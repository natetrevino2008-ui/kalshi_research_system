"""
config.py - Central configuration for the Kalshi NBA +EV engine.

Every tunable lives here. All money / probability / fee constants are
decimal.Decimal; nothing in the financial path uses binary floating point.

Environment variables (loaded from .env):
    KALSHI_KEY_ID / KALSHI_API_KEY_ID   API key id
    KALSHI_PRIVATE_KEY_PATH             path to the RSA private key (.pem), default private_key.pem
    KALSHI_PRIVATE_KEY                  (alternative) PEM text, \\n escapes allowed
    KALSHI_ENV                          "demo" (default) or "prod"
    KALSHI_BASE_URL                     (optional) overrides the URL implied by KALSHI_ENV
    ODDS_API_KEY                        The Odds API key
    KALSHI_ARM_LIVE                     must equal LIVE_ARM_PHRASE to allow real orders
    DAY_TIMEZONE                        (optional) IANA tz for the daily drawdown reset
"""
from __future__ import annotations

# =============================================================================
# SAFETY SWITCH - READ THIS FIRST
# -----------------------------------------------------------------------------
# DRY_RUN = True  -> orders are SIMULATED (written to the journal + paper ledger).
#                    No order endpoint is ever called.
# DRY_RUN = False -> real orders are sent ONLY IF the environment variable
#                    KALSHI_ARM_LIVE is also set to LIVE_ARM_PHRASE (two-key arming).
# =============================================================================
DRY_RUN = True
# =============================================================================

import os
from datetime import timedelta, timezone, tzinfo
from decimal import Decimal, getcontext
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass

getcontext().prec = 28

# ----------------------------------------------------------------------------
# Live-trading interlock
# ----------------------------------------------------------------------------
LIVE_ARM_ENV_VAR = "KALSHI_ARM_LIVE"
LIVE_ARM_PHRASE = "YES_I_WANT_REAL_ORDERS"


def live_trading_armed() -> bool:
    """True only when DRY_RUN is False AND the arming env var matches exactly."""
    return (not DRY_RUN) and os.getenv(LIVE_ARM_ENV_VAR, "") == LIVE_ARM_PHRASE


# ----------------------------------------------------------------------------
# Kalshi connection
# ----------------------------------------------------------------------------
KALSHI_ENV = os.getenv("KALSHI_ENV", "demo").strip().lower()
_KALSHI_BASE_URLS = {
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
    "prod": "https://external-api.kalshi.com/trade-api/v2",
}
KALSHI_BASE_URL = (os.getenv("KALSHI_BASE_URL") or _KALSHI_BASE_URLS.get(KALSHI_ENV) or "").rstrip("/")

# NBA moneyline ("game winner") series on Kalshi. Add more series tickers if needed.
KALSHI_SERIES_TICKERS = ("KXNBAGAME",)
KALSHI_MARKETS_PAGE_LIMIT = 1000
KALSHI_MAX_PAGES = 50


def kalshi_key_id() -> str | None:
    return os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_API_KEY_ID")


def kalshi_private_key_pem() -> bytes | None:
    """Return PEM bytes from a file path, or from PEM text in the environment."""
    raw_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "private_key.pem")
    try:
        candidate = Path(raw_path)
        if candidate.is_file():
            return candidate.read_bytes()
    except (OSError, ValueError):
        pass  # value was probably PEM text, not a path
    inline = os.getenv("KALSHI_PRIVATE_KEY")
    if inline:
        return inline.replace("\\n", "\n").encode("utf-8")
    if "BEGIN" in raw_path:  # legacy prototype pasted PEM text into the *_PATH variable
        return raw_path.replace("\\n", "\n").encode("utf-8")
    return None


# ----------------------------------------------------------------------------
# The Odds API
# ----------------------------------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "basketball_nba"
ODDS_MARKETS = "h2h"
# Quota cost per call = markets x regions, and every group of 10 bookmakers counts as
# one region. Naming up to 10 books explicitly (any region) costs 1 credit per scan and
# lets us request Pinnacle, which is not in the "us" region. Leave empty to fall back
# to ODDS_REGIONS.
ODDS_BOOKMAKERS = (
    "pinnacle",
    "lowvig",
    "betonlineag",
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill_us",
    "betrivers",
    "bovada",
    "mybookieag",
)
ODDS_REGIONS = "us,eu"  # used only when ODDS_BOOKMAKERS is empty (or as automatic fallback)
MIN_ODDS_CREDITS_REMAINING = 25  # main.py refuses to scan below this many credits

# ----------------------------------------------------------------------------
# Consensus pricing
# ----------------------------------------------------------------------------
DEVIG_METHOD = "power"  # "power" | "shin" | "multiplicative"
DEFAULT_BOOK_WEIGHT = Decimal("1.0")  # soft retail books
BOOK_WEIGHTS = {
    "pinnacle": Decimal("5.0"),
    "circa": Decimal("4.0"),
    "lowvig": Decimal("4.0"),
    "bookmaker": Decimal("4.0"),
    "betonlineag": Decimal("2.0"),
}
SHARP_BOOKS = frozenset({"pinnacle", "circa", "lowvig", "bookmaker"})
REQUIRE_SHARP_BOOK = True  # need >= 1 fresh sharp book in the consensus
MIN_BOOKS_FOR_CONSENSUS = 2

# Staleness gates (seconds). Per the spec both default to 60.
# MAX_ODDS_AGE_SECONDS      : age of OUR snapshot (time since the API response was received).
# MAX_BOOK_UPDATE_AGE_SECONDS: age of each bookmaker's own last_update timestamp.
# NOTE: pre-game lines often sit unchanged for minutes, so a strict 60s book-age gate
# will drop many books. Loosen MAX_BOOK_UPDATE_AGE_SECONDS if the journal shows mostly
# SKIP_STALE decisions.
MAX_ODDS_AGE_SECONDS = 60
MAX_BOOK_UPDATE_AGE_SECONDS = 60

# Pre-game only: the consensus is a pre-game number and goes stale the moment play starts.
SKIP_LIVE_GAMES = True
MIN_MINUTES_BEFORE_TIPOFF = 5

# ----------------------------------------------------------------------------
# Kalshi fee model (fee = ceil_to_cent(rate * contracts * P * (1 - P)))
# Source: Kalshi fee schedule. Per-series behaviour comes from the series object's
# fee_type / fee_multiplier fields.
# ----------------------------------------------------------------------------
TAKER_FEE_BASE_RATE = Decimal("0.07")
MAKER_FEE_BASE_RATE = Decimal("0.0175")  # fee_type == quadratic_with_maker_fees
COMBO_MAKER_FEE_BASE_RATE = Decimal("0.035")  # fee_type == quadratic_with_combo_maker_fees
# If the series lookup fails, assume maker fees DO apply (conservative).
FALLBACK_FEE_TYPE = "quadratic_with_maker_fees"
FALLBACK_FEE_MULTIPLIER = Decimal("1")

# ----------------------------------------------------------------------------
# Edge, spread, liquidity gates
# ----------------------------------------------------------------------------
MIN_NET_EDGE = Decimal("0.035")  # reject if Net_Edge < 3.5 percentage points
MAX_SPREAD = Decimal("0.04")  # reject if bid-ask spread > $0.04
TICK = Decimal("0.01")
MAKER_PRICE_IMPROVEMENT = Decimal("0.01")  # post at best_bid + $0.01 (never crossing the ask)
MIN_ASK_DEPTH_CONTRACTS = Decimal("5")  # market must show >= this many contracts at the best ask
DEPTH_COVERAGE_MULTIPLE = Decimal("1.0")  # taker orders need depth >= contracts * this
ALLOW_TAKER_ORDERS = True
# Maker-first: rest a post-only order unless the taker net edge is large enough that
# paying the fee to get filled immediately is worth it.
TAKER_OVERRIDE_MIN_EDGE = Decimal("0.06")

# ----------------------------------------------------------------------------
# Bankroll management & circuit breakers
# ----------------------------------------------------------------------------
KELLY_FRACTION = Decimal("0.25")  # quarter Kelly
MAX_BET_FRACTION = Decimal("0.05")  # single-bet risk cap as fraction of equity
MAX_BET_DOLLARS = Decimal("25.00")  # hard dollar cap per bet (whichever cap is lower wins)
MAX_TOTAL_EXPOSURE_FRACTION = Decimal("0.30")  # cap on all open positions + resting orders
MAX_DAILY_DRAWDOWN = Decimal("0.15")  # halt for the day at -15% of start-of-day equity
MAX_CONTRACTS_PER_ORDER = 1000
PAPER_BANKROLL = Decimal("100.00")  # simulated bankroll used when DRY_RUN is True
PAPER_LEDGER_TTL_HOURS = 36
# Kalshi's balance is "available for trading"; resting-order reserves are added back so
# that placing an order does not look like a loss. Set False if you verify on the demo
# account that balance already includes reserved funds.
COUNT_RESTING_ORDERS_IN_EQUITY = True

# ----------------------------------------------------------------------------
# HTTP resilience
# ----------------------------------------------------------------------------
HTTP_TIMEOUT_SECONDS = 15
HTTP_MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 300  # only used by `main.py --loop`

# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JOURNAL_CSV_PATH = DATA_DIR / "trade_journal.csv"
JOURNAL_JSONL_PATH = DATA_DIR / "trade_journal.jsonl"
RISK_STATE_PATH = DATA_DIR / "risk_state.json"
PAPER_LEDGER_PATH = DATA_DIR / "paper_ledger.json"

# ----------------------------------------------------------------------------
# Time zones (Windows needs the `tzdata` package for zoneinfo; we fall back safely)
# ----------------------------------------------------------------------------
TZ_FALLBACK_USED = False


def _get_timezone(name: str, fallback_utc_offset_hours: int) -> tzinfo:
    global TZ_FALLBACK_USED
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        TZ_FALLBACK_USED = True
        return timezone(timedelta(hours=fallback_utc_offset_hours))


GAME_TZ = _get_timezone("America/New_York", -5)  # Kalshi game tickers use the ET game date
DAY_TZ = _get_timezone(os.getenv("DAY_TIMEZONE", "America/Chicago"), -6)  # daily drawdown reset

# ----------------------------------------------------------------------------
# NBA team table: Kalshi 3-letter code -> (city, nickname). Odds API names are
# matched on the nickname ("Lakers" in "Los Angeles Lakers").
# ----------------------------------------------------------------------------
NBA_TEAMS = {
    "ATL": ("Atlanta", "Hawks"),
    "BOS": ("Boston", "Celtics"),
    "BKN": ("Brooklyn", "Nets"),
    "CHA": ("Charlotte", "Hornets"),
    "CHI": ("Chicago", "Bulls"),
    "CLE": ("Cleveland", "Cavaliers"),
    "DAL": ("Dallas", "Mavericks"),
    "DEN": ("Denver", "Nuggets"),
    "DET": ("Detroit", "Pistons"),
    "GSW": ("Golden State", "Warriors"),
    "HOU": ("Houston", "Rockets"),
    "IND": ("Indiana", "Pacers"),
    "LAC": ("Los Angeles", "Clippers"),
    "LAL": ("Los Angeles", "Lakers"),
    "MEM": ("Memphis", "Grizzlies"),
    "MIA": ("Miami", "Heat"),
    "MIL": ("Milwaukee", "Bucks"),
    "MIN": ("Minnesota", "Timberwolves"),
    "NOP": ("New Orleans", "Pelicans"),
    "NYK": ("New York", "Knicks"),
    "OKC": ("Oklahoma City", "Thunder"),
    "ORL": ("Orlando", "Magic"),
    "PHI": ("Philadelphia", "76ers"),
    "PHX": ("Phoenix", "Suns"),
    "POR": ("Portland", "Trail Blazers"),
    "SAC": ("Sacramento", "Kings"),
    "SAS": ("San Antonio", "Spurs"),
    "TOR": ("Toronto", "Raptors"),
    "UTA": ("Utah", "Jazz"),
    "WAS": ("Washington", "Wizards"),
}
NBA_TEAM_CODE_ALIASES = {
    "BRK": "BKN",
    "CHO": "CHA",
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "PHO": "PHX",
    "SA": "SAS",
    "UTH": "UTA",
    "WSH": "WAS",
}


def validate_config() -> list[str]:
    """Return a list of human-readable configuration problems (empty list == OK)."""
    problems: list[str] = []
    if not KALSHI_BASE_URL:
        problems.append("KALSHI_ENV must be 'demo' or 'prod' (or set KALSHI_BASE_URL).")
    if DEVIG_METHOD not in {"power", "shin", "multiplicative"}:
        problems.append("DEVIG_METHOD must be 'power', 'shin' or 'multiplicative'.")
    if not (Decimal("0") < KELLY_FRACTION <= Decimal("1")):
        problems.append("KELLY_FRACTION must be in (0, 1].")
    if not (Decimal("0") < MAX_BET_FRACTION <= Decimal("1")):
        problems.append("MAX_BET_FRACTION must be in (0, 1].")
    if not (Decimal("0") < MAX_DAILY_DRAWDOWN < Decimal("1")):
        problems.append("MAX_DAILY_DRAWDOWN must be in (0, 1).")
    if MIN_NET_EDGE < 0:
        problems.append("MIN_NET_EDGE must be >= 0.")
    if len(ODDS_BOOKMAKERS) > 10:
        problems.append("ODDS_BOOKMAKERS has more than 10 entries; each extra group of 10 costs another credit.")
    if not ODDS_API_KEY:
        problems.append("ODDS_API_KEY is missing from the environment / .env file.")
    return problems
