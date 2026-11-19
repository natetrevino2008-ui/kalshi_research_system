import csv
import os
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import streamlit as st

try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

import config
from execution_engine import PaperLedger, RiskManager, TradeJournal
from kalshi_client import KalshiClient
from main import build_mock_world, run_scan
from odds_client import OddsAPIClient

st.set_page_config(page_title="Kalshi NBA", page_icon="🏀", layout="centered")
config.DRY_RUN = True


def _num(val, default=0.0) -> float:
    try:
        return float(val) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def build_explanation(row: dict) -> str:
    team = row.get("team_yes") or "this team"
    side = str(row.get("side") or "yes").upper()
    fair_prob = _num(row.get("fair_prob_side")) * 100
    price_cents = _num(row.get("order_price")) * 100
    edge_pct = _num(row.get("net_edge")) * 100
    spread_cents = _num(row.get("spread")) * 100
    route = str(row.get("route") or "maker")

    raw_sharps = str(row.get("sharp_books") or "")
    sharps = ", ".join(b.capitalize() for b in raw_sharps.split("|") if b) or "sharp books"
    outcome = f"**{team}** to win" if side == "YES" else f"**{team}** to lose"
    fee_note = (
        "Placed as a resting maker order ($0 fees)."
        if route == "maker"
        else "Edge is large enough to buy immediately at the ask."
    )

    return (
        f"**Why this pick?** Sharp books ({sharps}) give {outcome} a **{fair_prob:.1f}%** true chance, "
        f"while Kalshi prices it at **{price_cents:.0f}¢** ({price_cents:.0f}% implied). "
        f"That leaves a **+{edge_pct:.1f}%** net edge with a tight **{spread_cents:.0f}¢** spread. "
        f"{fee_note}"
    )


def load_run_buys(csv_path, run_id: str) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("run_id") == run_id and r.get("decision") == "BUY"]


# --- Top Bar ---
top_l, top_r = st.columns([3, 1])
with top_l:
    st.title("🏀 Kalshi NBA")
with top_r:
    st.write("")
    with st.popover("⚙️ Tune", use_container_width=True):
        bankroll_val = st.number_input("Fake Bankroll ($)", 10.0, 50000.0, float(config.PAPER_BANKROLL), 25.0)
        min_edge_val = st.slider("Min Edge (%)", 0.5, 15.0, float(config.MIN_NET_EDGE * 100), 0.5)
        kelly_val = st.slider("Bet Size (Kelly)", 0.05, 1.0, float(config.KELLY_FRACTION), 0.05)
        reset_btn = st.button("Reset Paper Balance", use_container_width=True)

        config.PAPER_BANKROLL = Decimal(str(bankroll_val))
        config.MIN_NET_EDGE = Decimal(str(min_edge_val)) / Decimal("100")
        config.KELLY_FRACTION = Decimal(str(kelly_val))

# --- Mode Switcher ---
mode = st.segmented_control(
    "Mode",
    ["Offline Demo", "Live Prices (Paper)"],
    default="Offline Demo",
    label_visibility="collapsed",
) or "Offline Demo"

is_mock = mode == "Offline Demo"
config.DATA_DIR.mkdir(parents=True, exist_ok=True)

if is_mock:
    journal_paths = (config.DATA_DIR / "mock_trade_journal.csv", config.DATA_DIR / "mock_trade_journal.jsonl")
    risk_path = config.DATA_DIR / "mock_risk_state.json"
    ledger_path = config.DATA_DIR / "mock_paper_ledger.json"
else:
    journal_paths = (config.JOURNAL_CSV_PATH, config.JOURNAL_JSONL_PATH)
    risk_path = config.RISK_STATE_PATH
    ledger_path = config.PAPER_LEDGER_PATH

if reset_btn:
    for p in (ledger_path, risk_path, *journal_paths):
        p.unlink(missing_ok=True)
    st.session_state.pop("detailed_buys", None)
    st.toast("Paper balance reset!", icon="✨")

# --- Balance Bubble Card ---
ledger = PaperLedger(ledger_path)
acct = ledger.account(datetime.now(timezone.utc))

with st.container(border=True):
    b_left, b_right = st.columns([3, 1])
    b_left.caption("PAPER BALANCE")
    b_right.markdown(":blue-badge[Simulated]")
    st.metric("Available Cash", f"${acct.cash:,.2f}", label_visibility="collapsed")
    st.caption(f"Total Equity: **${acct.equity:,.2f}**  •  In Play: **${acct.exposure:,.2f}**")

# --- Scan Button ---
if st.button("Scan NBA Games", type="primary", use_container_width=True):
    if is_mock:
        for p in (ledger_path, risk_path):
            p.unlink(missing_ok=True)
        kalshi, fetch_snapshot = build_mock_world()
    else:
        problems = config.validate_config()
        if problems:
            st.error("Setup needed:\n- " + "\n- ".join(problems))
            st.stop()
        kalshi = KalshiClient()
        fetch_snapshot = OddsAPIClient().fetch_nba_odds

    with st.spinner("Scanning NBA matchups..."):
        journal = TradeJournal(*journal_paths)
        risk = RiskManager(risk_path)
        ledger = PaperLedger(ledger_path)
        run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger)
        st.session_state["detailed_buys"] = load_run_buys(journal.csv_path, journal.run_id)
        st.rerun()

# --- View Switcher ---
view = st.segmented_control(
    "View",
    ["✨ Best Picks", "📋 All Scanned"],
    default="✨ Best Picks",
    label_visibility="collapsed",
) or "✨ Best Picks"

if view == "✨ Best Picks":
    buys = st.session_state.get("detailed_buys")
    if buys is None:
        with st.container(border=True):
            st.markdown("🏀 **Ready to scan** — Tap **Scan NBA Games** above to find mispriced matchups.")
    elif not buys:
        with st.container(border=True):
            st.markdown("No strong edges right now. All games were priced fairly or had wide spreads.")
    else:
        for b in buys:
            edge_pct = _num(b.get("net_edge")) * 100
            price_val = _num(b.get("order_price"))
            stake_val = _num(b.get("stake"))
            contracts = int(_num(b.get("contracts")))
            side_label = str(b.get("side") or "yes").upper()
            team_name = b.get("team_yes") or b.get("kalshi_ticker")

            with st.container(border=True):
                c_title, c_badge = st.columns([3, 1])
                c_title.subheader(team_name)
                c_badge.markdown(f":green-badge[+{edge_pct:.1f}% Edge]")
                st.caption(b.get("kalshi_ticker", ""))

                c_order, c_stake = st.columns([3, 1])
                c_order.markdown(f"Buy **{side_label}** • {contracts} contracts @ **${price_val:.2f}**")
                c_stake.markdown(f"**${stake_val:.2f}**")

                st.info(build_explanation(b), icon="💡")
else:
    csv_file = journal_paths[0]
    if csv_file.exists() and csv_file.stat().st_size > 0:
        df = pd.read_csv(csv_file)
        clean_cols = [c for c in ["kalshi_ticker", "team_yes", "decision", "net_edge", "reason"] if c in df.columns]
        st.dataframe(df[clean_cols].tail(25), use_container_width=True, hide_index=True)
    else:
        st.caption("No scan history yet.")
