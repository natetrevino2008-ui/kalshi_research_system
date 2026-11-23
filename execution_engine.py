"""
execution_engine.py - Matching, gating, sizing, risk control, journaling, execution and settlement.

Two clearly separated stages:

  1. PROPOSE  (ExecutionEngine) - per-game decision pipeline, cheap gates first, API calls last:

        HALT_DRAWDOWN -> SKIP_LIVE -> SKIP_STALE -> SKIP_NO_ODDS / SKIP_CONSENSUS
          -> SKIP_DUPLICATE -> [prescreen -> order book -> SKIP_NO_BOOK -> SKIP_SPREAD -> PASS
             -> SKIP_LIQUIDITY -> SKIP_RISK -> BUY]

     A BUY here is a *pick* (an OrderPlan). Nothing is sent anywhere. Every evaluated market is
     written to the journal (CSV + JSONL).

  2. EXECUTE  (module functions) - a pick becomes a trade only through one of:
        execute_paper()      simulated fill, recorded in trade_ledger.csv (paper / demo)
        place_live_order()   REAL order on Kalshi; requires confirmed=True, re-prices against a fresh
                             order book, re-runs every risk check on the live account, then signs and
                             sends POST /portfolio/orders (resting post-only maker or fill-or-kill taker)

     settle_trades() later asks Kalshi which contracts settled and books the P&L.

Markets are only compared when the sportsbook quotes the EXACT same selection (same team / player,
same side, same point line - see math_engine.SelectionKey). Kalshi ladders (many spread / total /
prop thresholds per game) are matched to the lines books actually post; unmatched thresholds are
summarised in one journal row instead of hundreds.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import config
import math_engine as me
from kalshi_client import KalshiClient, MarketSpec, settlement_result, summarize_order
from odds_client import OddsEvent, OddsSnapshot

log = logging.getLogger("execution_engine")


# =============================================================================
# Decisions
# =============================================================================
class Decision(str, Enum):
    BUY = "BUY"  # a qualifying pick (proposal)
    PASS = "PASS"  # priced, but net edge below threshold
    SKIP_SPREAD = "SKIP_SPREAD"
    SKIP_LIQUIDITY = "SKIP_LIQUIDITY"
    SKIP_STALE = "SKIP_STALE"
    SKIP_NO_ODDS = "SKIP_NO_ODDS"
    SKIP_CONSENSUS = "SKIP_CONSENSUS"  # too few books on this exact line / no sharp book
    SKIP_NO_BOOK = "SKIP_NO_BOOK"  # empty, one-sided, locked or crossed order book
    SKIP_NO_MATCH = "SKIP_NO_MATCH"  # no Kalshi<->Odds API pairing (game, line or player)
    SKIP_LIVE = "SKIP_LIVE"
    SKIP_DUPLICATE = "SKIP_DUPLICATE"
    SKIP_RISK = "SKIP_RISK"
    SKIP_BUDGET = "SKIP_BUDGET"  # order-book call budget for this scan exhausted
    HALT_DRAWDOWN = "HALT_DRAWDOWN"
    ERROR = "ERROR"


# =============================================================================
# Audit journal (CSV + JSONL)
# =============================================================================
JOURNAL_FIELDS = [
    "ts_utc", "run_id", "decision", "reason", "route", "mode",
    "market_type", "market_label", "pick_label", "line",
    "event_id", "kalshi_event_ticker", "kalshi_ticker", "team_yes", "side",
    "commence_time", "minutes_to_tipoff",
    "fair_prob_yes_team", "fair_prob_side", "books_used", "sharp_books",
    "sharp_weight_share", "consensus_dispersion", "odds_age_seconds",
    "yes_bid", "yes_ask", "no_bid", "no_ask", "spread", "ask_depth",
    "order_price", "contracts", "stake", "fee_total", "fee_per_contract", "net_edge",
    "kelly_full", "kelly_applied", "binding_cap", "bankroll", "cash",
    "order_id", "client_order_id", "api_status", "api_response", "details_json",
]
_CSV_RESPONSE_LIMIT = 1500


def _json_default(value: Any) -> Any:
    if isinstance(value, (Decimal, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return "|".join(str(v) for v in value)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (dict,)):
        return json.dumps(value, default=_json_default)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _ensure_csv_header(path: Path, fields: list[str]) -> None:
    """Create the CSV with its header. A file with an older/different header is archived, never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open(encoding="utf-8", newline="") as handle:
            first = next(csv.reader(handle), [])
        if first == fields:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(f"{path.stem}.old-{stamp}{path.suffix}"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(fields)


class TradeJournal:
    """Append-only audit log. CSV is for spreadsheets; JSONL keeps the full API responses."""

    def __init__(self, csv_path: Path | None = None, jsonl_path: Path | None = None,
                 run_id: str | None = None, mode: str = "paper"):
        self.csv_path = Path(csv_path or config.JOURNAL_CSV_PATH)
        self.jsonl_path = Path(jsonl_path or config.JOURNAL_JSONL_PATH)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.mode = mode
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_csv_header(self.csv_path, JOURNAL_FIELDS)

    def log(self, decision: Decision, reason: str, rec: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "decision": decision.value,
            "reason": reason,
            "mode": self.mode,
            **rec,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")

        extras = {k: v for k, v in record.items() if k not in JOURNAL_FIELDS}
        row = dict(record)
        row["details_json"] = json.dumps(extras, default=_json_default) if extras else ""
        if row.get("api_response") is not None and not isinstance(row["api_response"], str):
            row["api_response"] = json.dumps(row["api_response"], default=_json_default)
        row["api_response"] = (row.get("api_response") or "")[:_CSV_RESPONSE_LIMIT]
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([_cell(row.get(name)) for name in JOURNAL_FIELDS])


# =============================================================================
# Kalshi <-> Odds API matching (game level; lines are matched per market in the engine)
# =============================================================================
GameKey = tuple[frozenset, date]


def event_from_ticker(market_ticker: str) -> str:
    return market_ticker.rsplit("-", 1)[0]


def game_code_of(ticker: str) -> str:
    parts = ticker.split("-")
    return parts[1] if len(parts) > 1 else ticker


def index_specs(specs: Iterable[MarketSpec]) -> dict[GameKey, list[MarketSpec]]:
    """Group every Kalshi market (moneyline, spread, total, props) of one game under one key."""
    out: dict[GameKey, list[MarketSpec]] = defaultdict(list)
    for spec in specs:
        out[(frozenset({spec.team_a, spec.team_b}), spec.game_date)].append(spec)
    return out


def _best_group(event: OddsEvent, index: dict[GameKey, list[MarketSpec]], used: set[GameKey]) -> GameKey | None:
    home, away = event.home_code, event.away_code
    if not home or not away:
        return None
    pair = frozenset({home, away})
    et_date = event.commence_time.astimezone(config.GAME_TZ).date()
    best: tuple[int, GameKey] | None = None
    for key in index:
        if key in used or key[0] != pair:
            continue
        delta = abs((key[1] - et_date).days)
        if delta <= 1 and (best is None or delta < best[0]):
            best = (delta, key)
    return best[1] if best else None


def prop_event_filter(specs: Iterable[MarketSpec]) -> Callable[[OddsEvent], bool]:
    """Predicate for the Odds API client: only games Kalshi lists player-prop markets for get the per-event call."""
    index = index_specs(s for s in specs if s.category == "prop")
    return lambda event: _best_group(event, index, set()) is not None


@dataclass
class MatchedGame:
    odds_event: OddsEvent
    game_code: str
    specs: list[MarketSpec]  # every Kalshi market for this game, all enabled categories

    @property
    def label(self) -> str:
        return f"{self.odds_event.away_team} @ {self.odds_event.home_team}"


def match_games(
    odds_events: list[OddsEvent], specs: list[MarketSpec]
) -> tuple[list[MatchedGame], list[OddsEvent], list[str]]:
    """Pair Odds API games with Kalshi games on (team pair, ET game date +-1 day)."""
    index = index_specs(specs)
    games: list[MatchedGame] = []
    unmatched_odds: list[OddsEvent] = []
    used: set[GameKey] = set()
    for event in odds_events:
        key = _best_group(event, index, used)
        if key is None:
            unmatched_odds.append(event)
            continue
        used.add(key)
        games.append(MatchedGame(event, index[key][0].game_code, index[key]))
    unmatched_kalshi = [f"{'/'.join(sorted(k[0]))} {k[1].isoformat()}" for k in index if k not in used]
    return games, unmatched_odds, unmatched_kalshi


# =============================================================================
# Account state, trade ledger, risk manager
# =============================================================================
@dataclass
class AccountState:
    equity: Decimal  # cash + position value (+ resting-order reserve) - the Kelly bankroll
    cash: Decimal  # available to trade
    exposure: Decimal  # cost of open positions + resting orders
    tickers: set[str]
    groups: set[str]  # exposure groups: ticker minus final segment (game for ML/spread/total, player for props)
    source: str  # "kalshi" or "paper"

    def has_exposure(self, ticker: str, group: str) -> bool:
        return ticker in self.tickers or group in self.groups

    def game_positions(self, game_code: str) -> int:
        return sum(1 for t in self.tickers if game_code_of(t) == game_code)

    def register(self, plan: "OrderPlan") -> None:
        self.cash -= plan.total_cost
        self.exposure += plan.total_cost
        self.tickers.add(plan.ticker)
        self.groups.add(plan.group)


def build_live_account(kalshi: KalshiClient) -> AccountState:
    balance = kalshi.get_balance()
    positions = kalshi.get_positions()
    resting = kalshi.get_resting_orders()
    position_exposure = sum((p["exposure"] for p in positions), me.ZERO)
    resting_notional = sum((o["notional"] for o in resting), me.ZERO)
    equity = balance["cash"] + balance["portfolio_value"]
    if config.COUNT_RESTING_ORDERS_IN_EQUITY:
        equity += resting_notional
    tickers = {p["ticker"] for p in positions} | {o["ticker"] for o in resting}
    return AccountState(
        equity=equity,
        cash=balance["cash"],
        exposure=position_exposure + resting_notional,
        tickers=tickers,
        groups={event_from_ticker(t) for t in tickers if t},
        source="kalshi",
    )


LEDGER_FIELDS = [
    # the columns you asked for
    "timestamp", "ticker", "market_type", "side", "price", "contracts", "
