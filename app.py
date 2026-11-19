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

# =============================================================================
# Clean Apple / iOS Bubble Styling
# =============================================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F5F7;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
        color: #1D1D1F;
    }
    header[data-testid="stHeader"], footer { visibility: hidden; }
    .block-container { max-width: 660px; padding-top: 2rem; padding-bottom: 3rem; }

    /* White rounded bubble cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 22px !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04) !important;
        padding: 6px 10px !important;
        margin-bottom: 10px !important;
    }

    /* Fix Tune button so text is clearly visible */
    div[data-testid="stPopover"] button {
        border-radius: 999px !important;
        background-color: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.14) !important;
        color: #1D1D1F !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPopover"] button * {
        color: #1D1D1F !important;
    }

    /* Apple Blue pill button */
    div.stButton > button[kind="primary"] {
        background-color: #007AFF !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25) !important;
    }

    .pill-blue {
        background: #E8F2FF; color: #007AFF; font-weight: 600;
        font-size: 12px; padding: 4px 12px; border-radius: 999px;
    }
    .pill-green {
        background: #E6F8EC; color: #248A3D; font-weight: 600;
        font-size: 12px; padding: 4px 12px; border-radius: 999px;
    }
    .why-box {
        background: #F5F5F7; border-radius: 14px; padding: 12px 14px;
        margin-top: 12px; font-size: 13px; line-height: 1.45; color: #3A3A3C;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    sharps = ", ".join(b.capitalize() for b in raw_sharps.split("|") if b) or "sharp sportsbooks"
    outcome_text = f"<b>{team}</b> to win" if side == "YES" else f"<b>{team}</b> to lose"
    route_note = (
        "Placed as a resting maker order to pay $0 in fees."
        if route == "maker"
        else "Edge is large enough to buy immediately at the ask."
    )

    return (
        f"Sharp books ({sharps}) give {outcome_text} a <b>{fair_prob:.1f}%</b> true chance, "
        f"while Kalshi is pricing it at <b>{price_cents:.0f}¢</b> ({price_cents:.0f}% implied). "
        f"That leaves a <b>+{edge_pct:.1f}%</b> net edge with a tight <b>{spread_cents:.0f}¢</b> spread. "
        f"{route_note}"
    )


# =============================================================================
# Top Header & Tune Popover
# =============================================================================
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("## 🏀 Kalshi NBA")
with c2:
    with st.popover("⚙️ Tune"):
        bankroll_val = st.number_input("Fake Bankroll ($)", 10.0, 50000.0, float(config.PAPER_BANKROLL), 25.0)
        min_edge_val = st.slider("Min Edge (%)", 0.5, 15.0, float(config.MIN_NET_EDGE * 100), 0.5)
        kelly_val = st.slider("Bet Size (Kelly)", 0.05, 1.0, float(config.KELLY_FRACTION), 0.05)
        reset_btn = st.button("Reset Paper Balance", use_container_width=True)

        config.PAPER_BANKROLL = Decimal(str(bankroll_val))
        config.MIN_NET_EDGE = Decimal(str(min_edge_val)) / Decimal("100")
        config.KELLY_FRACTION = Decimal(str(kelly_val))

# =============================================================================
# Mode Switcher
# =============================================================================
mode = st.radio(
    "Mode",
    ["Offline Demo", "Live Prices (Paper)"],
    horizontal=True,
    label_visibility="collapsed",
)
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

# =============================================================================
# Balance Bubble Card
# =============================================================================
ledger = PaperLedger(ledger_path)
acct = ledger.account(datetime.now(timezone.utc))

with st.container(border=True):
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:12px; font-weight:600; color:#86868B;">PAPER BALANCE</span>
            <span class="pill-blue">Simulated</span>
        </div>
        <div style="font-size:34px; font-weight:700; margin:4px 0;">${acct.cash:,.2f}</div>
        <div style="font-size:13px; color:#6E6E73;">
            Total Equity: <b>${acct.equity:,.2f}</b> &nbsp;•&nbsp; In Play: <b>${acct.exposure:,.2f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Scan Button
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

    with st.spinner("Scanning NBA matchups..."):
        journal = TradeJournal(*journal_paths)
        risk = RiskManager(risk_path)
        ledger = PaperLedger(ledger_path)
        run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger)

        detailed_buys = []
        if journal.csv_path.exists():
            df_run = pd.read_csv(journal.csv_path)
            df_run = df_run[(df_run["run_id"] == journal.run_id) & (df_run
