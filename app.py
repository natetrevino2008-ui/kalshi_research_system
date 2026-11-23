import hmac
import os
import time
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

try:
    import config
    from execution_engine import RiskManager, TradeJournal, TradeLedger, build_live_account, execute_paper, place_live_order
    from kalshi_client import KalshiClient
    from main import make_world, run_scan, settle_scope
    from odds_client import OddsAPIError
except ImportError as _exc:
    st.error("Sync error. Ensure all files match in repo.")
    st.stop()

st.set_page_config(page_title="TrueLine", page_icon="📈", layout="wide")

MODE_LABELS = list(config.MODE_LABELS.values())
LABEL_TO_MODE = {label: mode for mode, label in config.MODE_LABELS.items()}
MARKET_TAGS = {"moneyline": "Moneyline", "spread": "Spread", "total": "Total", "prop": "Player prop"}
STRETCH = {"use_container_width": True}

def _num(val, default=0.0):
    try: return float(val) if val not in ("", None) else default
    except: return default

top_l, top_r = st.columns([3, 1])
with top_l: st.title("TrueLine")

with top_r:
    with st.popover("⚙️ Settings", **STRETCH):
        bankroll_val = st.number_input("Bankroll ($)", 10.0, 50000.0, float(config.PAPER_BANKROLL), 25.0)
        min_edge_val = st.slider("Min Edge (%)", 0.5, 15.0, float(config.MIN_NET_EDGE * 100), 0.5)
        
        st.markdown("**Markets to scan**")
        cat_flags = {
            "moneyline": st.checkbox("Moneylines", value=True),
            "spread": st.checkbox("Spreads", value=True),
            "total": st.checkbox("Totals", value=True),
            "prop": st.checkbox("Player Props", value=False)
        }
        categories = tuple(k for k, on in cat_flags.items() if on)
        target_games = st.text_input("Target Games (e.g. LAL, BOS)", help="Filter games to save API credits")
        
        config.PAPER_BANKROLL = Decimal(str(bankroll_val))
        config.MIN_NET_EDGE = Decimal(str(min_edge_val)) / Decimal("100")
        config.ENABLED_CATEGORIES.update(cat_flags)

mode_label = st.segmented_control("Mode", MODE_LABELS, default="Offline Demo", label_visibility="collapsed") or "Offline Demo"
mode = LABEL_TO_MODE[mode_label]
is_demo, is_paper, is_live = mode == "demo", mode == "paper", mode == "live"

# --- Security Gate ---
if mode != "demo":
    pw = config.app_password()
    if pw and not st.session_state.get("auth_ok"):
        st.warning("Please authenticate to access Live Data / Real Money execution.")
        pwd_in = st.text_input("App Password", type="password")
        if pwd_in:
            if hmac.compare_digest(pwd_in.encode(), pw.encode()):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Invalid Password")
        st.stop()

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
paths = config.mode_paths(mode)
ledger = TradeLedger(paths.ledger_csv)

@st.cache_data(ttl=300, show_spinner=False)
def execute_scan_cached(mode, categories, target_games):
    kalshi, fetch_snapshot = make_world(mode)
    journal = TradeJournal(paths.journal_csv, paths.journal_jsonl, mode=mode)
    risk = RiskManager(paths.risk_state)
    return run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger, mode=mode,
                    categories=categories, target_games=target_games.split(",") if target_games else None,
                    auto_paper=(is_demo or is_paper), say=lambda _msg: None)

with st.container(border=True):
    b_left, b_right = st.columns([3, 1])
    acct = ledger.account(datetime.now(timezone.utc), mode)
    b_left.caption(f"{mode.upper()} BALANCE")
    st.metric("Available Cash", f"${acct.cash:,.2f}")
    st.caption(f"Total Equity: **${acct.equity:,.2f}** | Exposure: **${acct.exposure:,.2f}**")

col_scan, col_force = st.columns([3, 1])
force_refresh = col_force.checkbox("Force Fresh Scan", help="Bypass the 5-minute cache")

if col_scan.button("Scan NBA Games", type="primary", disabled=not categories, **STRETCH):
    if force_refresh:
        execute_scan_cached.clear()
    with st.spinner("Scanning markets..."):
        try:
            result = execute_scan_cached(mode, categories, target_games)
            st.session_state["picks"] = result.picks
        except Exception as e:
            st.error(f"Scan failed: {e}")

view = st.segmented_control("View", ["✨ Picks", "📊 Dashboard & Ledger"], default="✨ Picks", label_visibility="collapsed") or "✨ Picks"

if view == "✨ Picks":
    picks = st.session_state.get("picks")
    if picks:
        for pick in picks:
            with st.container(border=True):
                st.subheader(pick.plan.label)
                st.markdown(f"Buy **{pick.plan.side.upper()}** @ **${float(pick.plan.price):.2f}** (Edge: {float(pick.plan.net_edge)*100:.1f}%)")
    else:
        st.info("No active edges. Scan to discover.")

elif view == "📊 Dashboard & Ledger":
    st.warning("⚠️ **Storage Notice:** Streamlit Cloud resets disk space periodically. Download your CSVs to retain CLV/Ledger history.")
    
    if st.button("Refresh / Settle Trades"):
        settle_scope(ledger, (mode,))
        st.success("Ledger synced.")
        
    stats = ledger.stats(mode)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate", "—" if stats["win_rate"] is None else f"{float(stats['win_rate']) * 100:.1f}%")
    c2.metric("Total P&L", f"${float(stats['pnl']):+,.2f}")
    c3.metric("ROI", "—" if stats["roi"] is None else f"{float(stats['roi']) * 100:+.1f}%")
    
    rows = ledger.rows(mode)
    if rows:
        df = pd.DataFrame(rows)
        # Brier Score computation (Probability calibration)
        if 'devigged_prob' in df.columns and 'status' in df.columns:
            closed = df[df['status'].isin(['WON', 'LOST'])]
            if not closed.empty:
                closed['outcome'] = (closed['status'] == 'WON').astype(int)
                brier = ((pd.to_numeric(closed['devigged_prob']) - closed['outcome']) ** 2).mean()
                c4.metric("Brier Score", f"{brier:.3f}", help="Closer to 0.0 is better.")
                
        st.dataframe(df)
        st.download_button("Download CSV", ledger.path.read_bytes(), "trade_ledger.csv")
