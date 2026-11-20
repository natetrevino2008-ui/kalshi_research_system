import hmac
import os
import time
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import streamlit as st

# Streamlit secrets -> environment, BEFORE importing config (config reads the environment at import).
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

# The seven files (app, config, kalshi_client, odds_client, math_engine, execution_engine, main) are
# versioned together. If one in your repo is still an older copy, say so here: Streamlit Cloud redacts
# import errors, which makes this otherwise impossible to diagnose.
_REQUIRED_CONFIG = ("MODE_LABELS", "CATEGORIES", "mode_paths", "enabled_categories", "validate_config",
                    "live_credential_problems", "PAPER_DATA_BASE_URL", "TRADE_LEDGER_PATH", "estimate_credits")
try:
    import config
    _stale = [f"config.py is missing {name}" for name in _REQUIRED_CONFIG if not hasattr(config, name)]
    if _stale:
        raise ImportError("; ".join(_stale))
    from execution_engine import (
        RiskManager, TradeJournal, TradeLedger, build_live_account, execute_paper, place_live_order,
    )
    from kalshi_client import KalshiClient
    from main import make_world, run_scan, settle_scope
    from odds_client import OddsAPIError
except ImportError as _exc:
    st.error(
        "**TrueLine can't start because the files in your repo don't match each other.**\n\n"
        f"Details: `{_exc}`\n\n"
        "Replace **all seven** files together (`app.py`, `config.py`, `kalshi_client.py`, `odds_client.py`, "
        "`math_engine.py`, `execution_engine.py`, `main.py`), commit, then reboot the app. "
        "A file that still has the old code (for example an `execution_engine.py` without `TradeLedger`) causes this."
    )
    st.stop()

st.set_page_config(page_title="TrueLine", page_icon="📈", layout="centered")

MODE_LABELS = list(config.MODE_LABELS.values())  # Offline Demo | Live Paper Trade | Live Real Money
LABEL_TO_MODE = {label: mode for mode, label in config.MODE_LABELS.items()}
MARKET_TAGS = {"moneyline": "Moneyline", "spread": "Spread", "total": "Total", "prop": "Player prop"}

# Streamlit >= 1.50 replaced use_container_width=True with width="stretch".
try:
    _MAJOR_MINOR = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _MAJOR_MINOR = (1, 40)
STRETCH = {"width": "stretch"} if _MAJOR_MINOR >= (1, 50) else {"use_container_width": True}


def _num(val, default=0.0) -> float:
    try:
        return float(val) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def build_explanation(row: dict) -> str:
    label = row.get("pick_label") or row.get("team_yes") or "this pick"
    fair_prob = _num(row.get("fair_prob_side")) * 100
    price_cents = _num(row.get("order_price")) * 100
    edge_pct = _num(row.get("net_edge")) * 100
    spread_cents = _num(row.get("spread")) * 100
    route = str(row.get("route") or "maker")

    raw_sharps = row.get("sharp_books") or ()
    if isinstance(raw_sharps, str):
        raw_sharps = raw_sharps.split("|")
    sharps = ", ".join(str(b).capitalize() for b in raw_sharps if b)
    raw_books = row.get("books_used") or ()
    if isinstance(raw_books, str):
        raw_books = raw_books.split("|")
    n_books = len([b for b in raw_books if b])
    source = f"Sharp books ({sharps})" if sharps else f"{n_books} sportsbooks"
    line_note = " on the exact same line" if row.get("market_type") in ("spread", "total", "prop") else ""
    fee_note = (
        "Placed as a resting maker order ($0 fees)."
        if route == "maker"
        else "Edge is large enough to buy immediately at the ask."
    )
    return (
        f"**Why this pick?** {source} give **{label}** a **{fair_prob:.1f}%** true chance{line_note}, "
        f"while Kalshi prices it at **{price_cents:.0f}¢** ({price_cents:.0f}% implied). "
        f"That leaves a **+{edge_pct:.1f}%** net edge with a tight **{spread_cents:.0f}¢** spread. "
        f"{fee_note}"
    )


# --- Top Bar ---
top_l, top_r = st.columns([3, 1])
with top_l:
    st.title("TrueLine")
with top_r:
    st.write("")
    with st.popover("⚙️ Tune", **STRETCH):
        bankroll_val = st.number_input("Fake Bankroll ($) - paper/demo", 10.0, 50000.0, float(config.PAPER_BANKROLL), 25.0)
        min_edge_val = st.slider("Min Edge (%)", 0.5, 15.0, float(config.MIN_NET_EDGE * 100), 0.5)
        kelly_val = st.slider("Bet Size (Kelly)", 0.05, 1.0, float(config.KELLY_FRACTION), 0.05)
        st.markdown("**Markets to scan**")
        for _cat in config.CATEGORIES:  # defaults come from config.ENABLED_CATEGORIES, then the checkboxes own them
            st.session_state.setdefault(f"cat_{_cat}", bool(config.ENABLED_CATEGORIES.get(_cat)))
        st.session_state.setdefault("auto_paper", True)
        cat_flags = {
            "moneyline": st.checkbox("Moneylines", key="cat_moneyline"),
            "spread": st.checkbox("Spreads", key="cat_spread"),
            "total": st.checkbox("Totals", key="cat_total"),
            "prop": st.checkbox("Player Props", key="cat_prop",
                                help="Props use a separate Odds API call per game (4 markets x games)."),
        }
        categories = tuple(k for k, on in cat_flags.items() if on)
        regions = config.odds_regions_count()
        lines_cost = len(config.bulk_odds_markets(categories)) * regions
        props_cost = len(config.prop_odds_markets(categories)) * regions
        st.caption(
            f"Odds API credits per scan: ~{lines_cost} for lines"
            + (f" + {props_cost} per game with props (~{props_cost * 10} on a 10-game night)" if props_cost else "")
            + ". Moneylines/spreads/totals share one bulk call."
        )
        auto_paper = st.checkbox("Auto paper-trade qualifying picks", key="auto_paper")
        reset_btn = st.button("Reset Paper Balance", **STRETCH)

        config.PAPER_BANKROLL = Decimal(str(bankroll_val))
        config.MIN_NET_EDGE = Decimal(str(min_edge_val)) / Decimal("100")
        config.KELLY_FRACTION = Decimal(str(kelly_val))
        config.ENABLED_CATEGORIES.update(cat_flags)

# --- Mode Switcher ---
mode_label = st.segmented_control(
    "Mode", MODE_LABELS, default="Offline Demo", label_visibility="collapsed", key="mode_switch",
) or "Offline Demo"  # deselecting falls back to the safe, no-money mode
mode = LABEL_TO_MODE[mode_label]
is_demo, is_paper, is_live = mode == "demo", mode == "paper", mode == "live"

if st.session_state.get("active_mode") != mode:  # switching modes drops picks and re-locks real money
    st.session_state["active_mode"] = mode
    for key in ("picks", "scan_meta", "pick_results", "live_account", "live_enabled", "live_pw_ok", "reset_armed_at"):
        st.session_state.pop(key, None)

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
paths = config.mode_paths(mode)
ledger = TradeLedger(paths.ledger_csv)
sim_mode = "demo" if is_demo else "paper"  # which simulated account "Reset" / the paper balance refer to

# --- Reset Paper Balance (needs a second click within 10 seconds; the whole ledger is backed up first) ---
if reset_btn:
    armed_at = st.session_state.get("reset_armed_at", 0.0)
    if time.time() - armed_at <= 10:
        sim_paths = config.mode_paths(sim_mode)
        sim_ledger = TradeLedger(sim_paths.ledger_csv)
        if is_demo:
            for p in (sim_paths.ledger_csv, sim_paths.risk_state, sim_paths.journal_csv, sim_paths.journal_jsonl):
                p.unlink(missing_ok=True)
        else:
            sim_ledger.reset_mode("paper")  # real-money rows are never touched
            RiskManager(sim_paths.risk_state).reset_source("paper")
        st.session_state.pop("reset_armed_at", None)
        st.session_state.pop("picks", None)
        st.session_state.pop("pick_results", None)
        st.toast("Paper balance reset (ledger backed up to data/backups).", icon="✨")
        ledger = TradeLedger(paths.ledger_csv)
    else:
        st.session_state["reset_armed_at"] = time.time()
        st.toast("Click Reset Paper Balance again within 10 seconds to confirm.", icon="⚠️")


# --- Live Real Money gate ---
def live_gate() -> bool:
    """Shows the real-money panel. True only when order buttons may be used."""
    prod = config.is_prod()
    if prod:
        st.error(f"**PRODUCTION Kalshi environment ({config.KALSHI_ENV}).** Orders placed here spend REAL money.")
    else:
        st.warning(f"Kalshi **{config.KALSHI_ENV}** environment: real order flow, but play money. "
                   "Set KALSHI_ENV=prod to trade real funds.")
    problems = config.live_credential_problems()
    if problems:
        st.error("Setup needed:\n- " + "\n- ".join(problems))
        return False
    password = config.app_password()
    if prod and config.REQUIRE_APP_PASSWORD_FOR_PROD and not password:
        st.error("Real-money orders on production are locked until you set **APP_PASSWORD** in your secrets / .env. "
                 "This stops anyone who finds a public link to this app from using your Kalshi account.")
        return False
    if password and not st.session_state.get("live_pw_ok"):
        entered = st.text_input("App password", type="password", key="live_pw_input")
        if entered and hmac.compare_digest(entered.encode(), password.encode()):
            st.session_state["live_pw_ok"] = True
            st.rerun()
        elif entered:
            st.error("Wrong password.")
        return False
    enabled = st.checkbox(
        "Enable Live Real Money", key="live_enabled",
        help="Unlocks the per-trade confirmation buttons. Nothing is sent until you press one of them.")
    known = st.session_state.get("live_account")
    cap = min(config.MAX_BET_FRACTION * known["equity"], config.MAX_BET_DOLLARS) if known else config.MAX_BET_DOLLARS
    st.caption(f"Each order is re-priced on a fresh order book when you confirm. Per-bet cap: up to ${cap:,.2f}.")
    if st.button("Refresh Kalshi balance", key="refresh_live_balance"):
        try:
            acct = build_live_account(KalshiClient())
            st.session_state["live_account"] = {"equity": acct.equity, "cash": acct.cash, "exposure": acct.exposure,
                                                "at": datetime.now(timezone.utc)}
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read your Kalshi account: {exc}")
    return bool(enabled)


live_ready = False
if is_demo:
    st.caption("Synthetic games and quotes. No network, no money.")
elif is_paper:
    st.caption("Real Kalshi (production market data, read-only) + sportsbook prices, simulated money. "
               "Every trade is saved to the Trade Ledger.")
else:
    live_ready = live_gate()

# --- Balance Bubble Card ---
with st.container(border=True):
    b_left, b_right = st.columns([3, 1])
    if is_live:
        b_left.caption("KALSHI BALANCE")
        b_right.markdown(":red-badge[REAL]" if config.is_prod() else ":orange-badge[Kalshi demo]")
        live_acct = st.session_state.get("live_account")
        if live_acct:
            st.metric("Available Cash", f"${live_acct['cash']:,.2f}", label_visibility="collapsed")
            st.caption(f"Total Equity: **${live_acct['equity']:,.2f}**  •  In Play: **${live_acct['exposure']:,.2f}**  •  "
                       f"as of {live_acct['at']:%H:%M:%S} UTC")
        else:
            st.markdown("Balance not loaded yet. Scan, or press *Refresh Kalshi balance*.")
    else:
        acct = ledger.account(datetime.now(timezone.utc), sim_mode)
        b_left.caption("DEMO BALANCE" if is_demo else "PAPER BALANCE")
        b_right.markdown(":gray-badge[Demo]" if is_demo else ":blue-badge[Simulated]")
        st.metric("Available Cash", f"${acct.cash:,.2f}", label_visibility="collapsed")
        st.caption(f"Total Equity: **${acct.equity:,.2f}**  •  In Play: **${acct.exposure:,.2f}**")

# --- Scan Button ---
if st.button("Scan NBA Games", type="primary", disabled=not categories, **STRETCH):
    if is_demo:  # every demo scan starts from a clean sandbox
        for p in (paths.ledger_csv, paths.risk_state):
            p.unlink(missing_ok=True)
        ledger = TradeLedger(paths.ledger_csv)
    else:
        problems = config.validate_config(categories, mode)
        if problems:
            st.error("Setup needed:\n- " + "\n- ".join(problems))
            st.stop()
    result = None
    try:
        with st.spinner("Scanning NBA matchups..."):
            kalshi, fetch_snapshot = make_world(mode)
            journal = TradeJournal(paths.journal_csv, paths.journal_jsonl, mode=mode)
            risk = RiskManager(paths.risk_state)
            result = run_scan(
                kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger, mode=mode, categories=categories,
                auto_paper=(auto_paper or is_demo) and not is_live, say=lambda _msg: None,
            )
    except OddsAPIError as exc:
        st.error(f"Odds API problem: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scan failed: {exc}")
    if result is not None:
        st.session_state["picks"] = result.picks
        st.session_state["pick_results"] = {
            pick.plan.client_order_id: (ok, "Paper trade recorded automatically." if ok else f"Not recorded: {why}")
            for pick, ok, why in result.executed
        }
        snap = result.snapshot
        st.session_state["scan_meta"] = {
            "run_id": result.run_id, "at": datetime.now(timezone.utc), "games": result.games,
            "kalshi_markets": result.kalshi_markets, "unresolved": result.unresolved,
            "book_calls": result.engine.book_calls, "credits_used": snap.scan_cost,
            "credits_left": snap.credits_remaining, "warnings": list(snap.warnings) + list(result.kalshi_errors),
        }
        if is_live and result.account is not None:
            st.session_state["live_account"] = {"equity": result.account.equity, "cash": result.account.cash,
                                                "exposure": result.account.exposure, "at": datetime.now(timezone.utc)}
        st.rerun()

meta = st.session_state.get("scan_meta")
if meta:
    st.caption(
        f"Last scan {meta['at']:%H:%M:%S} UTC • {meta['games']} games • {meta['kalshi_markets']} Kalshi markets • "
        f"{meta['book_calls']} order-book calls • Odds API credits: {meta['credits_used']} used"
        + (f", {meta['credits_left']} left" if meta["credits_left"] is not None else "")
    )
    for warning in meta["warnings"]:
        st.warning(warning)
    if meta["unresolved"]:
        st.caption(f"{meta['unresolved']} Kalshi markets could not be matched safely (see All Scanned for reasons).")

# --- View Switcher ---
view = st.segmented_control(
    "View", ["✨ Best Picks", "📋 All Scanned", "🧾 Trade Ledger"], default="✨ Best Picks",
    label_visibility="collapsed", key="view_switch",
) or "✨ Best Picks"


def render_pick(pick, results: dict) -> None:
    plan, rec = pick.plan, pick.rec
    cid = plan.client_order_id
    edge_pct = float(plan.net_edge) * 100
    age = (datetime.now(timezone.utc) - plan.created_at).total_seconds()
    with st.container(border=True):
        c_title, c_badge = st.columns([3, 1])
        c_title.subheader(plan.label)
        c_badge.markdown(f":green-badge[+{edge_pct:.1f}% Edge]")
        st.caption(f"{MARKET_TAGS.get(plan.market_type, plan.market_type)} • {plan.ticker}")

        c_order, c_stake = st.columns([3, 1])
        how = "resting maker order" if plan.route == "maker" else "immediate taker order"
        c_order.markdown(f"Buy **{plan.side.upper()}** • {plan.contracts} contracts @ **${float(plan.price):.2f}** ({how})")
        c_stake.markdown(f"**${float(plan.total_cost):.2f}**")
        st.info(build_explanation(rec), icon="💡")
        if not is_demo and age > config.LIVE_PLAN_MAX_AGE_SECONDS and cid not in results:
            st.caption(f"Scanned {age / 60:.0f} min ago - prices may have moved. Rescan for fresh quotes.")

        outcome = results.get(cid)
        if outcome is not None:
            ok, message = outcome
            (st.success if ok else st.error)(message, icon="✅" if ok else "⚠️")
            return
        if is_live:
            stale = age > config.LIVE_PLAN_MAX_AGE_SECONDS
            if stale:
                st.caption(f"This pick is {age:.0f}s old. Rescan before trading real money.")
            elif not live_ready:
                st.caption("Tick **Enable Live Real Money** above to unlock the confirm button.")
            confirm = st.button(
                f"Confirm & place REAL order • max loss ${float(plan.total_cost):.2f}", key=f"live_{cid}",
                type="primary", disabled=(stale or not live_ready), **STRETCH)
            if confirm and live_ready and st.session_state.get("live_enabled"):
                with st.spinner("Re-pricing and sending order to Kalshi..."):
                    kalshi = KalshiClient()
                    outcome = place_live_order(
                        pick, kalshi=kalshi, journal=TradeJournal(paths.journal_csv, paths.journal_jsonl, run_id=pick.run_id, mode=mode),
                        risk=RiskManager(paths.risk_state), ledger=ledger, confirmed=True)
                    try:
                        refreshed = build_live_account(kalshi)
                        st.session_state["live_account"] = {"equity": refreshed.equity, "cash": refreshed.cash,
                                                            "exposure": refreshed.exposure, "at": datetime.now(timezone.utc)}
                    except Exception:  # noqa: BLE001 - balance refresh is best-effort
                        pass
                results[cid] = (outcome.ok, outcome.message)
                st.rerun()
        else:
            if st.button("Place paper bet", key=f"paper_{cid}", **STRETCH):
                ok, message = execute_paper(
                    pick, ledger=ledger, mode=sim_mode, risk=RiskManager(paths.risk_state),
                    journal=TradeJournal(paths.journal_csv, paths.journal_jsonl, run_id=pick.run_id, mode=mode))
                results[cid] = (ok, "Paper trade recorded." if ok else f"Not recorded: {message}")
                st.rerun()


if view == "✨ Best Picks":
    picks = st.session_state.get("picks")
    if picks is None:
        with st.container(border=True):
            st.markdown("**Ready to scan** — tap **Scan NBA Games** above to find mispriced lines.")
    elif not picks:
        with st.container(border=True):
            st.markdown("No strong edges right now. Everything was priced fairly, had wide spreads, or no sportsbook "
                        "quoted the exact same line.")
    else:
        results = st.session_state.setdefault("pick_results", {})
        for pick in picks:
            render_pick(pick, results)

elif view == "📋 All Scanned":
    csv_file = paths.journal_csv
    if csv_file.exists() and csv_file.stat().st_size > 0:
        df = pd.read_csv(csv_file, dtype=str).fillna("")
        if df.empty:
            st.caption("No scan history yet.")
        else:
            run_id = (st.session_state.get("scan_meta") or {}).get("run_id") or df["run_id"].iloc[-1]
            df = df[df["run_id"] == run_id].copy()
            df["Edge %"] = pd.to_numeric(df["net_edge"], errors="coerce") * 100
            df["Market"] = df["market_type"].map(MARKET_TAGS).fillna(df["market_type"])
            df["Selection"] = df["pick_label"].where(df["pick_label"] != "", df["team_yes"])
            decisions = sorted(df["decision"].unique())
            chosen = st.multiselect("Show decisions", decisions, default=decisions, key="decision_filter")
            df = df[df["decision"].isin(chosen)]
            df["_buy"] = (df["decision"] != "BUY")
            df = df.sort_values(["_buy", "Edge %"], ascending=[True, False], na_position="last")
            st.dataframe(
                df[["Market", "Selection", "decision", "Edge %", "reason", "kalshi_ticker"]].head(300),
                hide_index=True, column_config={"Edge %": st.column_config.NumberColumn(format="%.2f"),
                                                "decision": "Decision", "reason": "Why", "kalshi_ticker": "Ticker"},
                **STRETCH)
            st.caption(f"Run {run_id}: {len(df)} rows shown (max 300). Full detail: {csv_file.name}")
    else:
        st.caption("No scan history yet.")

else:
    # ------------------------------------------------------------------ Trade Ledger
    scope = st.segmented_control("Show", ["This mode", "All trades"], default="This mode", key="ledger_scope") or "This mode"
    if is_demo:
        ledger_mode = "demo"
    else:  # paper and real trades share one file; "All trades" shows both
        ledger_mode = None if scope == "All trades" else mode

    if st.button("Refresh / Settle Trades", key="settle_btn", **STRETCH):
        with st.spinner("Checking Kalshi for settled contracts..."):
            try:
                modes = ("demo",) if is_demo else (("paper", "live") if ledger_mode is None else (ledger_mode,))
                report = settle_scope(ledger, modes)
                if is_live or (ledger_mode is None and "live" in modes):
                    try:
                        refreshed = build_live_account(KalshiClient())
                        st.session_state["live_account"] = {
                            "equity": refreshed.equity, "cash": refreshed.cash, "exposure": refreshed.exposure,
                            "at": datetime.now(timezone.utc)}
                    except Exception as exc:  # noqa: BLE001
                        if is_live:
                            st.warning(f"Trades updated, but the Kalshi balance could not be refreshed: {exc}")
                st.success(f"Checked {report.checked} open trade(s): {report.settled} settled, "
                           f"{report.updated} updated, {report.canceled} canceled.")
                for error in report.errors:
                    st.warning(error)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not refresh trades: {exc}")

    stats = ledger.stats(ledger_mode)
    m1, m2 = st.columns(2)
    m1.metric("Win rate", "—" if stats["win_rate"] is None else f"{float(stats['win_rate']) * 100:.1f}%")
    m2.metric("Total P&L", f"${float(stats['pnl']):+,.2f}")
    m3, m4 = st.columns(2)
    m3.metric("ROI", "—" if stats["roi"] is None else f"{float(stats['roi']) * 100:+.1f}%")
    m4.metric("Open trades", f"{stats['open']}")
    st.caption(f"{stats['wins']} of {stats['settled']} settled trades won • ${float(stats['staked']):,.2f} staked on "
               f"settled trades • ${float(stats['open_stake']):,.2f} still at risk in {stats['open']} open trade(s)")

    rows = ledger.rows(ledger_mode)
    if not rows:
        st.caption("No trades yet. Qualifying picks you paper-trade (or real orders you confirm) are saved here.")
    else:
        table = pd.DataFrame(rows[::-1])
        for col in ("price", "contracts", "devigged_prob", "edge_pct", "stake", "pnl", "filled"):
            table[col] = pd.to_numeric(table[col], errors="coerce")
        table["devigged_prob"] = table["devigged_prob"] * 100
        table["mode"] = table["mode"].map({"paper": "Paper", "live": "Real", "demo": "Demo"}).fillna(table["mode"])
        table["market_type"] = table["market_type"].map(MARKET_TAGS).fillna(table["market_type"])
        st.dataframe(
            table[["timestamp", "mode", "status", "market_type", "label", "side", "price", "contracts",
                   "devigged_prob", "edge_pct", "stake", "pnl", "ticker"]],
            hide_index=True,
            column_config={
                "timestamp": "Time (UTC)", "mode": "Mode", "status": "Status", "market_type": "Market",
                "label": "Selection", "side": "Side", "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "contracts": "Contracts", "devigged_prob": st.column_config.NumberColumn("Fair prob %", format="%.1f"),
                "edge_pct": st.column_config.NumberColumn("Edge %", format="%.2f"),
                "stake": st.column_config.NumberColumn("Stake", format="$%.2f"),
                "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"), "ticker": "Ticker",
            },
            **STRETCH)
        st.download_button("Download ledger CSV", data=ledger.path.read_bytes(), file_name="trade_ledger.csv",
                           mime="text/csv", key="dl_ledger")
    st.caption("Saved to data/trade_ledger.csv. Paper fills are assumed at your limit price, so resting maker orders "
               "look better here than they would in real life. On hosted Streamlit the disk can reset when the app "
               "restarts, so download the CSV regularly.")
