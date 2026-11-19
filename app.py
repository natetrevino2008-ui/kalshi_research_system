import os
import pandas as pd
import streamlit as st

# Load Streamlit Cloud Secrets into environment variables before importing config
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

st.set_page_config(page_title="Kalshi NBA +EV Scanner", layout="wide")
st.title("Kalshi NBA +EV Scanner")
st.caption(f"Mode: {'DRY_RUN (Simulated)' if config.DRY_RUN else 'LIVE'} | Env: {config.KALSHI_ENV}")

mode = st.radio(
    "Data Source",
    ["Offline Mock Data (No API Keys Needed)", "Live Odds & Kalshi Order Books (DRY_RUN)"],
    horizontal=True,
)

if st.button("Run Scan", type="primary"):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if mode.startswith("Offline"):
        kalshi, fetch_snapshot = build_mock_world()
        journal_paths = (config.DATA_DIR / "mock_trade_journal.csv", config.DATA_DIR / "mock_trade_journal.jsonl")
        risk_path = config.DATA_DIR / "mock_risk_state.json"
        ledger_path = config.DATA_DIR / "mock_paper_ledger.json"
        for f in (risk_path, ledger_path):
            f.unlink(missing_ok=True)
    else:
        problems = config.validate_config()
        if problems:
            st.error("Configuration issues:\n- " + "\n- ".join(problems))
            st.stop()
        kalshi = KalshiClient()
        fetch_snapshot = OddsAPIClient().fetch_nba_odds
        journal_paths = (config.JOURNAL_CSV_PATH, config.JOURNAL_JSONL_PATH)
        risk_path = config.RISK_STATE_PATH
        ledger_path = config.PAPER_LEDGER_PATH

    with st.spinner("Scanning markets..."):
        journal = TradeJournal(*journal_paths)
        risk = RiskManager(risk_path)
        ledger = PaperLedger(ledger_path)
        result = run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger)

    engine = result["engine"]
    st.subheader("Decision Counts")
    st.json(dict(engine.counts))

    st.subheader("Simulated Orders")
    if engine.buys:
        st.dataframe(pd.DataFrame(engine.buys), use_container_width=True)
    else:
        st.info("No +EV trades passed all gates on this scan.")

    st.subheader("Full Decision Journal")
    if journal.csv_path.exists():
        st.dataframe(pd.read_csv(journal.csv_path), use_container_width=True)
