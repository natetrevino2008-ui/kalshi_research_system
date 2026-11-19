import os
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import streamlit as st

# 1. Load Streamlit Cloud Secrets into env vars before importing config
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

st.set_page_config(
    page_title="Kalshi NBA +EV Research Dashboard",
    page_icon="🏀",
    layout="wide",
)

# Force simulation safety lock in the UI
config.DRY_RUN = True

# =============================================================================
# Sidebar: Simulation & Strategy Controls
# =============================================================================
with st.sidebar:
    st.header("⚙️ Strategy Controls")
    st.caption("Adjust parameters before running a simulated scan.")

    data_mode = st.radio(
        "Data Source",
        ["Offline Mock Games", "Live Odds + Kalshi Books"],
        help="Offline Mock uses built-in synthetic NBA scenarios. Live fetches real market prices in DRY_RUN mode.",
    )

    st.divider()
    st.subheader("Bankroll & Sizing")
    paper_bankroll = st.number_input(
        "Starting Paper Bankroll ($)",
        min_value=10.0,
        max_value=100000.0,
        value=float(config.PAPER_BANKROLL),
        step=25.0,
    )
    kelly_fraction = st.slider(
        "Kelly Fraction",
        min_value=0.05,
        max_value=1.00,
        value=float(config.KELLY_FRACTION),
        step=0.05,
        help="0.25 = Quarter Kelly (recommended to reduce variance).",
    )

    st.divider()
    st.subheader("Entry Gates")
    min_edge_pct = st.slider(
        "Minimum Net Edge (%)",
        min_value=0.5,
        max_value=15.0,
        value=float(config.MIN_NET_EDGE * 100),
        step=0.5,
    )
    max_spread = st.slider(
        "Max Bid-Ask Spread ($)",
        min_value=0.01,
        max_value=0.20,
        value=float(config.MAX_SPREAD),
        step=0.01,
    )
    devig_method = st.selectbox(
        "Devigging Formula",
        ["power", "shin", "multiplicative"],
        index=["power", "shin", "multiplicative"].index(config.DEVIG_METHOD),
    )
    max_book_age = st.slider(
        "Max Sportsbook Quote Age (sec)",
        min_value=30,
        max_value=900,
        value=int(config.MAX_BOOK_UPDATE_AGE_SECONDS),
        step=30,
    )

    # Apply UI controls to config using exact Decimal types
    config.PAPER_BANKROLL = Decimal(str(paper_bankroll))
    config.KELLY_FRACTION = Decimal(str(kelly_fraction))
    config.MIN_NET_EDGE = Decimal(str(min_edge_pct)) / Decimal("100")
    config.MAX_SPREAD = Decimal(str(max_spread))
    config.DEVIG_METHOD = devig_method
    config.MAX_BOOK_UPDATE_AGE_SECONDS = int(max_book_age)

    st.divider()
    reset_clicked = st.button("🔄 Reset Paper Ledger", use_container_width=True)

# =============================================================================
# File paths for selected mode
# =============================================================================
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
is_mock = data_mode.startswith("Offline")

if is_mock:
    journal_paths = (config.DATA_DIR / "mock_trade_journal.csv", config.DATA_DIR / "mock_trade_journal.jsonl")
    risk_path = config.DATA_DIR / "mock_risk_state.json"
    ledger_path = config.DATA_DIR / "mock_paper_ledger.json"
else:
    journal_paths = (config.JOURNAL_CSV_PATH, config.JOURNAL_JSONL_PATH)
    risk_path = config.RISK_STATE_PATH
    ledger_path = config.PAPER_LEDGER_PATH

if reset_clicked:
    for path in (ledger_path, risk_path, *journal_paths):
        path.unlink(missing_ok=True)
    st.toast("Paper ledger and trade history cleared!", icon="✅")

# =============================================================================
# Header & Top KPI Metrics
# =============================================================================
st.title("🏀 Kalshi NBA +EV Research Terminal")
st.caption(
    f"Mode: **SIMULATION (DRY_RUN)** | Kalshi Env: `{config.KALSHI_ENV}` | "
    f"Devig: `{config.DEVIG_METHOD}` | Min Edge: `{min_edge_pct:.1f}%`"
)

ledger = PaperLedger(ledger_path)
acct = ledger.account(datetime.now(timezone.utc))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Paper Equity", f"${acct.equity:,.2f}")
m2.metric("Available Cash", f"${acct.cash:,.2f}")
m3.metric("Open Simulated Exposure", f"${acct.exposure:,.2f}")
m4.metric("Active Simulated Games", len(acct.events))

st.divider()

# =============================================================================
# Action Bar
# =============================================================================
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("▶️ Run Market Scan", type="primary", use_container_width=True)
with col_info:
    if is_mock:
        st.info("Running against 8 built-in synthetic NBA matchups (no API keys required).")
    else:
        st.warning("Live Data Mode: fetches real Odds API lines & Kalshi order books, but only places simulated paper trades.")

if run_clicked:
    # Fresh mock sandbox per scan so you can repeatedly test slider changes
    if is_mock:
        for path in (ledger_path, risk_path):
            path.unlink(missing_ok=True)
        kalshi, fetch_snapshot = build_mock_world()
    else:
        problems = config.validate_config()
        if problems:
            st.error("Missing or invalid configuration:\n- " + "\n- ".join(problems))
            st.stop()
        kalshi = KalshiClient()
        fetch_snapshot = OddsAPIClient().fetch_nba_odds

    with st.spinner("Fetching odds, checking Kalshi order books, and calculating +EV..."):
        journal = TradeJournal(*journal_paths)
        risk = RiskManager(risk_path)
        ledger = PaperLedger(ledger_path)
        result = run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger)
        st.session_state["last_counts"] = dict(result["engine"].counts)
        st.session_state["last_buys"] = result["engine"].buys
        st.session_state["last_run_id"] = journal.run_id
        st.rerun()

# =============================================================================
# Main Dashboard Tabs
# =============================================================================
tab_signals, tab_positions, tab_journal = st.tabs([
    "🎯 Latest Scan Signals",
    "📒 Paper Positions",
    "📊 Full Decision Log",
])

with tab_signals:
    buys = st.session_state.get("last_buys", [])
    counts = st.session_state.get("last_counts", {})

    if not counts:
        st.write("Click **▶️ Run Market Scan** above to evaluate NBA markets.")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Triggered +EV Trades")
            if buys:
                df_buys = pd.DataFrame([
                    {
                        "Ticker": b["ticker"],
                        "Side": b["side"].upper(),
                        "Route": b["route"],
                        "Contracts": b["contracts"],
                        "Limit Price": f"${float(b['price']):.2f}",
                        "Net Edge": f"{float(b['net_edge']) * 100:.2f}%",
                        "Max Stake": f"${float(b['stake']):.2f}",
                    }
                    for b in buys
                ])
                st.dataframe(df_buys, use_container_width=True, hide_index=True)
            else:
                st.info("No markets met your minimum edge, spread, and liquidity thresholds on this scan.")

        with right:
            st.subheader("Gate Breakdown")
            df_counts = pd.DataFrame(
                list(counts.items()), columns=["Decision", "Markets"]
            ).set_index("Decision")
            st.bar_chart(df_counts)

with tab_positions:
    st.subheader("Simulated Open Orders & Positions")
    if ledger.entries:
        df_pos = pd.DataFrame(ledger.entries)
        st.dataframe(df_pos, use_container_width=True, hide_index=True)
    else:
        st.write("No simulated positions recorded yet.")

with tab_journal:
    csv_file = journal_paths[0]
    if csv_file.exists() and csv_file.stat().st_size > 0:
        df_journal = pd.read_csv(csv_file)
        decisions = ["ALL"] + sorted(df_journal["decision"].dropna().unique().tolist())
        selected_dec = st.selectbox("Filter by Gate Decision", decisions)
        if selected_dec != "ALL":
            df_journal = df_journal[df_journal["decision"] == selected_dec]

        st.dataframe(df_journal, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Journal CSV",
            data=csv_file.read_bytes(),
            file_name=csv_file.name,
            mime="text/csv",
        )
    else:
        st.write("Run a scan to populate the audit journal.")
