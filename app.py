import os
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import streamlit as st

# Load Streamlit Secrets if present
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

# =============================================================================
# Apple / iOS Bubble Styling
# =============================================================================
st.markdown(
    """
    <style>
    /* Soft iOS System Gray background & San Francisco font stack */
    .stApp {
        background-color: #F5F5F7;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
        color: #1D1D1F;
    }
    /* Hide Streamlit header/footer clutter */
    header[data-testid="stHeader"], footer { visibility: hidden; }
    .block-container { max-width: 680px; padding-top: 2rem; padding-bottom: 3rem; }

    /* Floating white iOS bubble cards */
    .ios-card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.03);
    }
    .ios-label {
        font-size: 12px;
        font-weight: 600;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .ios-big-num {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1D1D1F;
        line-height: 1.1;
    }
    .ios-sub {
        font-size: 13px;
        color: #86868B;
        margin-top: 6px;
    }

    /* Pill badges */
    .pill-blue {
        display: inline-block;
        background: #E8F2FF;
        color: #007AFF;
        font-weight: 600;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 999px;
    }
    .pill-green {
        display: inline-block;
        background: #E6F8EC;
        color: #248A3D;
        font-weight: 600;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 999px;
    }

    /* Apple Blue pill buttons */
    div.stButton > button[kind="primary"] {
        background-color: #007AFF;
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 14px rgba(0, 122, 255, 0.28);
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0066D6;
        transform: translateY(-1px);
    }
    div[data-testid="stPopover"] > button {
        border-radius: 999px;
        background: #FFFFFF;
        border: 1px solid rgba(0,0,0,0.08);
        color: #1D1D1F;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Top Bar: Title + Minimal Settings Popover (No Sidebar!)
# =============================================================================
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("## 🏀 Kalshi NBA")
with top_right:
    with st.popover("⚙️ Tune"):
        st.markdown("**Paper Settings**")
        bankroll_val = st.number_input("Fake Bankroll ($)", 10.0, 50000.0, float(config.PAPER_BANKROLL), 25.0)
        min_edge_val = st.slider("Min Edge (%)", 0.5, 15.0, float(config.MIN_NET_EDGE * 100), 0.5)
        kelly_val = st.slider("Bet Size (Kelly)", 0.05, 1.0, float(config.KELLY_FRACTION), 0.05)
        reset_btn = st.button("Reset Paper Balance", use_container_width=True)

        config.PAPER_BANKROLL = Decimal(str(bankroll_val))
        config.MIN_NET_EDGE = Decimal(str(min_edge_val)) / Decimal("100")
        config.KELLY_FRACTION = Decimal(str(kelly_val))

# =============================================================================
# iOS Segmented Mode Switcher
# =============================================================================
mode = st.segmented_control(
    "Data Source",
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
    st.toast("Paper balance reset!", icon="✨")

# =============================================================================
# Balance Bubble Card (Apple Wallet Style)
# =============================================================================
ledger = PaperLedger(ledger_path)
acct = ledger.account(datetime.now(timezone.utc))

st.markdown(
    f"""
    <div class="ios-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="ios-label">Paper Balance</span>
            <span class="pill-blue">Simulated</span>
        </div>
        <div class="ios-big-num">${acct.cash:,.2f}</div>
        <div class="ios-sub">
            Total Equity: <b>${acct.equity:,.2f}</b> &nbsp;•&nbsp; In Play: <b>${acct.exposure:,.2f}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Big Pill Scan Button
# =============================================================================
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

    with st.spinner("Checking lines..."):
        journal = TradeJournal(*journal_paths)
        risk = RiskManager(risk_path)
        ledger = PaperLedger(ledger_path)
        result = run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger)
        st.session_state["buys"] = result["engine"].buys
        st.session_state["counts"] = dict(result["engine"].counts)
        st.rerun()

# =============================================================================
# View Switcher: Picks vs All Games
# =============================================================================
view = st.segmented_control(
    "View",
    ["✨ Best Picks", "📋 All Scanned"],
    default="✨ Best Picks",
    label_visibility="collapsed",
) or "✨ Best Picks"

if view == "✨ Best Picks":
    buys = st.session_state.get("buys")
    if buys is None:
        st.markdown(
            """
            <div class="ios-card" style="text-align:center; padding: 32px 20px;">
                <div style="font-size: 28px; margin-bottom: 8px;">🏀</div>
                <div style="font-weight: 600; font-size: 16px;">Ready to scan</div>
                <div class="ios-sub">Tap <b>Scan NBA Games</b> above to find mispriced matchups.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not buys:
        st.markdown(
            """
            <div class="ios-card" style="text-align:center; padding: 28px 20px;">
                <div style="font-weight: 600; font-size: 16px;">No strong edges right now</div>
                <div class="ios-sub">All games were either priced fairly or had wide spreads.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for b in buys:
            edge_pct = float(b["net_edge"]) * 100
            price_val = float(b["price"])
            stake_val = float(b["stake"])
            side_label = b["side"].upper()
            st.markdown(
                f"""
                <div class="ios-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:700; font-size:17px;">{b['ticker']}</span>
                        <span class="pill-green">+{edge_pct:.1f}% Edge</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="ios-sub" style="margin:0;">
                            Buy <b>{side_label}</b> • {b['contracts']} contracts @ <b>${price_val:.2f}</b>
                        </span>
                        <span style="font-weight:600; font-size:15px;">${stake_val:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    csv_file = journal_paths[0]
    if csv_file.exists() and csv_file.stat().st_size > 0:
        df = pd.read_csv(csv_file)
        clean_cols = [c for c in ["kalshi_ticker", "team_yes", "decision", "net_edge", "reason"] if c in df.columns]
        st.dataframe(df[clean_cols].tail(25), use_container_width=True, hide_index=True)
    else:
        st.caption("No scan history yet.")
