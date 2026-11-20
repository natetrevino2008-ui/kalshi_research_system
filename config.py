"""
config.py - Central configuration for TrueLine (Kalshi NBA +EV engine).

Every tunable lives here. All money / probability / fee constants are
decimal.Decimal; nothing in the financial path uses binary floating point.

Environment variables (loaded from .env or Streamlit secrets):
    KALSHI_API_KEY / KALSHI_KEY_ID      API key id (either name works)
    KALSHI_PRIVATE_KEY                  PEM text of the RSA private key (\\n escapes allowed)
    KALSHI_PRIVATE_KEY_PATH             (alternative) path to the .pem file, default private_key.pem
    KALSHI_ENV                          where REAL ORDERS go: "demo" (default, play money) or "prod" (REAL money)
    KALSHI_PAPER_DATA_ENV               market data used by Live Paper Trade: "prod" (default, read-only)
    KALSHI_BASE_URL                     (optional) overrides the URL implied by KALSHI_ENV
    ODDS_API_KEY                        The Odds API key
    APP_PASSWORD                        required to unlock Live Real Money against KALSHI_ENV=prod
    KALSHI_ARM_LIVE                     CLI only: must equal LIVE_ARM_PHRASE to allow real orders
    DAY_TIMEZONE                        (optional) IANA tz for the daily drawdown reset

Trading modes (see MODE_LABELS):
    demo   synthetic data, simulated money, separate mock_* files
    paper  real Kalshi + Odds API prices, simulated money, saved to trade_ledger.csv
    live   real Kalshi orders (RSA-PSS signed). The Streamlit app requires an explicit
           "Enable Live Real Money" checkbox plus a per-trade confirmation click.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone, tzinfo
from decimal import Decimal, getcontext
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass

getcontext().prec = 28

APP_NAME = "TrueLine"

# ----------------------------------------------------------------------------
# Trading modes
# ----------------------------------------------------------------------------
MODES = ("demo", "paper", "live")
MODE_LABELS = {"demo": "Offline Demo", "paper": "Live Paper Trade", "live": "Live Real Money"}

# CLI-only interlock for `python main.py --mode live` (the Streamlit app uses its checkbox +
# per-trade confirmation instead).
LIVE_ARM_ENV_VAR = "KALSHI_ARM_LIVE"
LIVE_ARM_PHRASE = "YES_I_WANT_REAL_ORDERS"


def live_trading_armed() -> bool:
    """True only when the arming env var matches exactly (CLI live mode)."""
    return os.getenv(LIVE_ARM_ENV_VAR, "") == LIVE_ARM_PHRASE


# ----------------------------------------------------------------------------
# Kalshi connection
# ----------------------------------------------------------------------------
KALSHI_ENV = os.getenv("KALSHI_ENV", "demo").strip().lower()
_KALSHI_BASE_URLS = {
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
    "prod": "https://external-api.kalshi.com/trade-api/v2",
}
KALSHI_BASE_URL = (os.getenv("KALSHI_BASE_URL") or _KALSHI_BASE_URLS.get(KALSHI_ENV) or "").rstrip("/")
# Live Paper Trade needs REAL market prices. They are public and read-only, so paper mode always reads
# the production exchange (KALSHI_PAPER_DATA_ENV, default "prod") no matter what KALSHI_ENV is. KALSHI_ENV
# only decides where REAL ORDERS go (Live Real Money).
PAPER_DATA_ENV = os.getenv("KALSHI_PAPER_DATA_ENV", "prod").strip().lower()
PAPER_DATA_BASE_URL = (_KALSHI_BASE_URLS.get(PAPER_DATA_ENV) or _KALSHI_BASE_URLS["prod"]).rstrip("/")
KALSHI_MARKETS_PAGE_LIMIT = 1000
KALSHI_MAX_PAGES = 50


def is_prod() -> bool:
    """True when orders would hit Kalshi's production (real money) environment."""
    return "demo" not in KALSHI_BASE_URL and bool(KALSHI_BASE_URL)


def kalshi_key_id() -> str | None:
    return os.getenv("KALSHI_API_KEY") or os.getenv("KALSHI_KEY_ID") or os.getenv("KALSHI_API_KEY_ID")


def kalshi_private_key_pem() -> bytes | None:
    """Return PEM bytes from KALSHI_PRIVATE_KEY text, or from a file path."""
    inline = os.getenv("KALSHI_PRIVATE_KEY")
    if inline and "BEGIN" in inline:
        return inline.replace("\\n", "\n").encode("utf-8")
    raw_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "private_key.pem")
    try:
        candidate = Path(raw_path)
        if candidate.is_file():
            return candidate.read_bytes()
    except (OSError, ValueError):
        pass  # value was probably PEM text, not a path
    if "BEGIN" in raw_path:  # legacy prototype pasted PEM text into the *_PATH variable
        return raw_path.replace("\\n", "\n").encode("utf-8")
    return None


def app_password() -> str | None:
    return os.getenv("APP_PASSWORD") or None


# When True, Live Real Money against the PRODUCTION environment is refused unless APP_PASSWORD is
# configured (and entered). Protects a publicly hosted app from strangers placing orders.
REQUIRE_APP_PASSWORD_FOR_PROD = True

# ----------------------------------------------------------------------------
# The Odds API
# ----------------------------------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "basketball_nba"
# Quota cost per call = markets x regions, and every group of 10 bookmakers counts as one
# region. Naming up to 10 books explicitly (any region) costs 1 region per call and lets us
# request Pinnacle, which is not in the "us" region. Leave empty to fall back to ODDS_REGIONS.
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
MIN_ODDS_CREDITS_REMAINING = 25  # never spend credits below this floor (checked before each call)
ODDS_MAX_HOURS_AHEAD = 36  # bulk /odds only asks for games starting within this window
PROPS_MAX_HOURS_AHEAD = 24  # per-event prop calls (4 credits each) only for games this close

# ----------------------------------------------------------------------------
# Market categories (what TrueLine can scan)
#
#   category   Kalshi series                         Odds API market(s)            endpoint
#   moneyline  KXNBAGAME                             h2h                           bulk /odds
#   spread     KXNBASPREAD                           spreads                       bulk /odds
#   total      KXNBATOTAL                            totals                        bulk /odds
#   prop       KXNBAPTS/REB/AST/3PT                  player_points/rebounds/...    per-event /events/{id}/odds
#
# Moneylines, spreads and totals share ONE bulk call (cost = markets x regions). Player props
# are only available per event, so they cost (markets x regions) credits PER GAME and are fetched
# only for games that Kalshi actually lists prop markets for.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketCategory:
    key: str
    label: str
    odds_markets: tuple[str, ...]
    per_event_endpoint: bool
    min_books: int  # fresh books quoting the EXACT same line needed for a consensus
    require_sharp: bool  # need >= 1 fresh sharp book among them
    edge_buffer: Decimal  # added on top of MIN_NET_EDGE (props are noisier)
    max_spread: Decimal  # reject Kalshi books wider than this


PROP_STATS = {
    "points": {"series": "KXNBAPTS", "odds_market": "player_points", "label": "Points"},
    "rebounds": {"series": "KXNBAREB", "odds_market": "player_rebounds", "label": "Rebounds"},
    "assists": {"series": "KXNBAAST", "odds_market": "player_assists", "label": "Assists"},
    "threes": {"series": "KXNBA3PT", "odds_market": "player_threes", "label": "3-Pointers"},
}
ENABLED_PROP_STATS = {"points": True, "rebounds": True, "assists": True, "threes": True}

CATEGORIES = {
    "moneyline": MarketCategory("moneyline", "Moneylines", ("h2h",), False, 2, True, Decimal("0"), Decimal("0.04")),
    "spread": MarketCategory("spread", "Spreads", ("spreads",), False, 2, True, Decimal("0"), Decimal("0.04")),
    "total": MarketCategory("total", "Totals", ("totals",), False, 2, True, Decimal("0"), Decimal("0.04")),
    # Props: fewer sharp books post them, so ask for 3 books on the exact line, do not insist on a
    # sharp one, and demand an extra 1.5 points of edge. Thinner Kalshi books -> wider spread allowed.
    "prop": MarketCategory(
        "prop", "Player Props", tuple(s["odds_market"] for s in PROP_STATS.values()), True,
        3, False, Decimal("0.015"), Decimal("0.06"),
    ),
}
_FIXED_SERIES = {"moneyline": ("KXNBAGAME",), "spread": ("KXNBASPREAD",), "total": ("KXNBATOTAL",)}

# Default toggles (the Streamlit "Tune" popover overrides these per session). Props are off by
# default because they are the expensive ones in Odds API credits.
ENABLED_CATEGORIES = {"moneyline": True, "spread": True, "total": True, "prop": False}


def enabled_categories() -> tuple[str, ...]:
    return tuple(key for key in CATEGORIES if ENABLED_CATEGORIES.get(key))


def enabled_prop_stats() -> tuple[str, ...]:
    return tuple(stat for stat in PROP_STATS if ENABLED_PROP_STATS.get(stat))


def kalshi_series_for(category: str) -> tuple[str, ...]:
    if category == "prop":
        return tuple(PROP_STATS[stat]["series"] for stat in enabled_prop_stats())
    return _FIXED_SERIES.get(category, ())


def series_category(series: str) -> str | None:
    for category, series_list in _FIXED_SERIES.items():
        if series in series_list:
            return category
    if any(series == info["series"] for info in PROP_STATS.values()):
        return "prop"
    return None


def series_stat(series: str) -> str | None:
    for stat, info in PROP_STATS.items():
        if info["series"] == series:
            return stat
    return None


def bulk_odds_markets(categories) -> list[str]:
    """Markets served by the single bulk /odds endpoint (h2h, spreads, totals only)."""
    wanted = {"moneyline": "h2h", "spread": "spreads", "total": "totals"}
    return [market for category, market in wanted.items() if category in categories]


def prop_odds_markets(categories) -> list[str]:
    """Player-prop markets (per-event endpoint only), limited to enabled stats."""
    if "prop" not in categories:
        return []
    return [PROP_STATS[stat]["odds_market"] for stat in enabled_prop_stats()]


def odds_regions_count() -> int:
    if ODDS_BOOKMAKERS:
        return max(1, -(-len(ODDS_BOOKMAKERS) // 10))  # ceil(n / 10)
    return max(1, len([r for r in ODDS_REGIONS.split(",") if r.strip()]))


def estimate_credits(categories, n_games: int = 10) -> int:
    """Upper-bound Odds API credits for one scan: 1 bulk call + (prop markets x games) per-event calls."""
    regions = odds_regions_count()
    bulk = len(bulk_odds_markets(categories)) * regions
    props = len(prop_odds_markets(categories)) * regions * n_games
    return bulk + props


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

# Staleness gates (seconds).
# MAX_ODDS_AGE_SECONDS       : age of OUR snapshot (time since the API response was received).
#                              Raised from 60 to 120 because a multi-market scan now makes many
#                              more Kalshi order-book calls after the odds are fetched.
# MAX_BOOK_UPDATE_AGE_SECONDS: age of each bookmaker's own last_update timestamp.
# NOTE: pre-game lines often sit unchanged for minutes, so a strict 60s book-age gate will drop
# many books. Loosen MAX_BOOK_UPDATE_AGE_SECONDS if the journal shows mostly SKIP_STALE decisions.
MAX_ODDS_AGE_SECONDS = 120
MAX_BOOK_UPDATE_AGE_SECONDS = 60

# Pre-game only: the consensus is a pre-game number and goes stale the moment play starts.
SKIP_LIVE_GAMES = True
MIN_MINUTES_BEFORE_TIPOFF = 5

# ----------------------------------------------------------------------------
# Kalshi fee model (fee = ceil_to_cent(rate * contracts * P * (1 - P)))
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
MIN_NET_EDGE = Decimal("0.035")  # reject if Net_Edge < 3.5 percentage points (+ category buffer)
MAX_SPREAD = Decimal("0.04")  # default Kalshi spread cap; categories can override (see CATEGORIES)
TICK = Decimal("0.01")
MAKER_PRICE_IMPROVEMENT = Decimal("0.01")  # post at best_bid + $0.01 (never crossing the ask)
MIN_ASK_DEPTH_CONTRACTS = Decimal("5")  # market must show >= this many contracts at the best ask
DEPTH_COVERAGE_MULTIPLE = Decimal("1.0")  # taker orders need depth >= contracts * this
ALLOW_TAKER_ORDERS = True
# Maker-first: rest a post-only order unless the taker net edge is large enough that
# paying the fee to get filled immediately is worth it.
TAKER_OVERRIDE_MIN_EDGE = Decimal("0.06")

# Order-book call budget. Prop ladders can have hundreds of Kalshi markets per game, so markets are
# (a) matched to an exact sportsbook line first and (b) screened with the quotes already in the
# /markets listing; only markets within PRESCREEN_SLACK of the edge threshold get a book fetch.
PRESCREEN_SLACK = Decimal("0.02")
MAX_ORDERBOOK_CALLS_PER_SCAN = 150

# ----------------------------------------------------------------------------
# Bankroll management & circuit breakers
# ----------------------------------------------------------------------------
KELLY_FRACTION = Decimal("0.25")  # quarter Kelly
MAX_BET_FRACTION = Decimal("0.05")  # single-bet risk cap as fraction of equity
MAX_BET_DOLLARS = Decimal("25.00")  # hard dollar cap per bet (whichever cap is lower wins)
MAX_TOTAL_EXPOSURE_FRACTION = Decimal("0.30")  # cap on all open positions + resting orders
MAX_POSITIONS_PER_GAME = 4  # moneyline + spread + total + props on one game are correlated
MAX_DAILY_DRAWDOWN = Decimal("0.15")  # halt for the day at -15% of start-of-day equity
MAX_CONTRACTS_PER_ORDER = 1000
PAPER_BANKROLL = Decimal("100.00")  # simulated starting bankroll (paper + demo)
# Kalshi's balance is "available for trading"; resting-order reserves are added back so
# that placing an order does not look like a loss. Set False if you verify on the demo
# account that balance already includes reserved funds.
COUNT_RESTING_ORDERS_IN_EQUITY = True
# A real-money pick is re-priced against a fresh order book when you press Confirm; picks older
# than this are refused ("rescan first").
LIVE_PLAN_MAX_AGE_SECONDS = 120

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
TRADE_LEDGER_PATH = DATA_DIR / "trade_ledger.csv"  # every paper + real trade, persistent


@dataclass(frozen=True)
class ModePaths:
    journal_csv: Path
    journal_jsonl: Path
    risk_state: Path
    ledger_csv: Path


def mode_paths(mode: str) -> ModePaths:
    """Demo is fully sandboxed in mock_* files; paper and live share the real files."""
    if mode == "demo":
        return ModePaths(
            DATA_DIR / "mock_trade_journal.csv", DATA_DIR / "mock_trade_journal.jsonl",
            DATA_DIR / "mock_risk_state.json", DATA_DIR / "mock_trade_ledger.csv",
        )
    return ModePaths(JOURNAL_CSV_PATH, JOURNAL_JSONL_PATH, RISK_STATE_PATH, TRADE_LEDGER_PATH)


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
# Kalshi shortens the two Los Angeles teams to "Los Angeles L" / "Los Angeles C" in market text.
TEAM_TEXT_ALIASES = {
    "los angeles l": "LAL",
    "los angeles c": "LAC",
    "la lakers": "LAL",
    "la clippers": "LAC",
}


def validate_config(categories=None, mode: str = "paper") -> list[str]:
    """Return a list of human-readable configuration problems (empty list == OK)."""
    problems: list[str] = []
    categories = tuple(categories) if categories is not None else enabled_categories()
    if not categories:
        problems.append("Select at least one market type to scan.")
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
        problems.append("ODDS_API_KEY is missing from the environment / .env file / Streamlit secrets.")
    if mode == "live":
        problems.extend(live_credential_problems())
    return problems


def live_credential_problems() -> list[str]:
    problems: list[str] = []
    if not kalshi_key_id():
        problems.append("KALSHI_API_KEY is missing (set it in Streamlit secrets or .env).")
    if not kalshi_private_key_pem():
        problems.append("KALSHI_PRIVATE_KEY is missing (PEM text in secrets/.env, or KALSHI_PRIVATE_KEY_PATH).")
    return problems
