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
# Apple / iOS Bubble Styling (Forces clean light cards regardless of browser theme)
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
    .block-container { max-width: 680px; padding-top: 2rem; padding-bottom: 3rem; }

    /* Floating white iOS bubble cards */
    .ios-card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.04);
        color: #1D1D1F;
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
        color: #6E6E73;
        margin-top: 6px;
    }

    /* Inner explanation bubble under each pick */
    .why-box {
        background: #F5F5F7;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 14px;
        font-size: 13px;
        line-height: 1.45;
        color: #3A3A3C;
    }
    .why-title {
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 2px;
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

    /* Apple Blue primary button */
    div.stButton > button[kind="primary"] {
        background-color: #007AFF !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 14px rgba(0, 122, 255, 0.28) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0066D6 !important;
    }

    /* Fix Tune popover button contrast */
    div[data-testid="stPopover"] > button {
        border-radius: 999px !important;
        background-color: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.12) !important;
        color: #1D1D1F !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid="stPopover"] > button p {
        color: #1D1D1F !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_explanation(row: dict) -> str:
    """Turn raw scan numbers into a simple human-readable explanation."""
    team = row.get("team_yes") or "this team"
    side = str(row.get("side") or "yes").upper()
    fair_prob = float(row.get("fair_prob_side") or 0) * 100
    price_cents = float(row.get("order_price") or 0) * 100
    edge_pct = float(row.get("net_edge") or 0) * 100
    spread_cents = float(row.get("spread") or 0) * 100
    route = str(row.get("route") or "maker")

    raw_sharps = str(row.get("sharp_books") or "")
    sharps = ", ".join(b.capitalize() for b in raw_sharps.split("|") if b) or "sharp sportsbooks"
    outcome_text = f"<b>{team}</b> to win" if side == "YES" else f"<b>{team}</b> to lose"

    route_note = (
        "Placed as a resting maker order to avoid taker fees."
        if route == "maker"
        else "Edge is big enough to buy immediately at the ask."
    )

    return (
        f"Sharp books ({sharps}) give {outcome_text} a <b>{fair_prob:.1f}%</b> true chance, "
        f"but Kalshi is pricing it at just <b>{price_cents:.0f}¢</b> ({price_cents:.0f}% implied). "
        f"That leaves a <b>+{edge_pct:.1f}%</b> net edge with a tight <b>{spread_cents:.0f}¢</b> spread. "
        f"{route_note}"
    )


# =============================================================================
# Top Bar: Title + Minimal Settings Popover
# =============================================================================
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown("## 🏀 Kalshi NBA")
with top_right:
    with st.popover("⚙️ Tune"):
        st.markdown("**Paper Settings**")
        bankroll_val = st
