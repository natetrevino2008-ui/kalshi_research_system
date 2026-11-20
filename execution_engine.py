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
    "timestamp", "ticker", "market_type", "side", "price", "contracts", "devigged_prob", "edge_pct", "status",
    # bookkeeping
    "trade_id", "mode", "env", "label", "event_ticker", "route", "stake", "fee_total", "filled",
    "order_id", "client_order_id", "settled_at", "result", "payout", "pnl", "note",
]
OPEN_STATUSES = ("OPEN", "RESTING")
CLOSED_STATUSES = ("WON", "LOST")  # VOID refunds are tracked but excluded from win rate / ROI


def _dec(row: dict, key: str, default: Decimal = me.ZERO) -> Decimal:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return Decimal(str(value))
    except ArithmeticError:
        return default


class TradeLedger:
    """
    Persistent record of every paper and real trade (CSV). Paper cash is derived from it:

        cash   = PAPER_BANKROLL - sum(stake of every trade) + sum(payout of settled trades)
        equity = cash + stake still at risk  (= bankroll + realised P&L)

    A trade stays OPEN (and keeps counting as exposure) until Kalshi reports the market settled.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.TRADE_LEDGER_PATH)
        _ensure_csv_header(self.path, LEDGER_FIELDS)

    # ------------------------------------------------------------------ storage
    def rows(self, mode: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [r for r in rows if mode is None or r.get("mode") == mode]

    def _write_all(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(self.path)

    def append(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, extrasaction="ignore").writerow(
                {name: _cell(row.get(name)) for name in LEDGER_FIELDS})

    def update_many(self, updates: dict[str, dict]) -> None:
        if not updates:
            return
        rows = self.rows()
        for row in rows:
            row.update({k: _cell(v) for k, v in updates.get(row.get("trade_id", ""), {}).items()})
        self._write_all(rows)

    def reset_mode(self, mode: str) -> Path | None:
        """Delete one mode's trades (paper reset). The whole ledger is backed up first; real trades are untouched."""
        rows = self.rows()
        if not rows:
            return None
        backup = self.path.parent / "backups" / f"{self.path.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        self._write_all([r for r in rows if r.get("mode") != mode])
        return backup

    # ------------------------------------------------------------------ derived state
    def account(self, now: datetime, mode: str) -> AccountState:
        rows = [r for r in self.rows(mode) if r.get("status") in (*OPEN_STATUSES, *CLOSED_STATUSES, "VOID")]
        spent = sum((_dec(r, "stake") for r in rows), me.ZERO)
        paid = sum((_dec(r, "payout") for r in rows if r.get("status") not in OPEN_STATUSES), me.ZERO)
        open_rows = [r for r in rows if r.get("status") in OPEN_STATUSES]
        at_risk = sum((_dec(r, "stake") for r in open_rows), me.ZERO)
        cash = config.PAPER_BANKROLL - spent + paid
        return AccountState(
            equity=cash + at_risk,
            cash=cash,
            exposure=at_risk,
            tickers={r["ticker"] for r in open_rows},
            groups={event_from_ticker(r["ticker"]) for r in open_rows},
            source="paper",
        )

    def stats(self, mode: str | None = None) -> dict[str, Any]:
        rows = self.rows(mode)
        closed = [r for r in rows if r.get("status") in CLOSED_STATUSES]
        wins = [r for r in closed if r.get("status") == "WON"]
        open_rows = [r for r in rows if r.get("status") in OPEN_STATUSES]
        pnl = sum((_dec(r, "pnl") for r in closed), me.ZERO)
        staked = sum((_dec(r, "stake") for r in closed), me.ZERO)
        return {
            "trades": len(rows),
            "open": len(open_rows),
            "open_stake": sum((_dec(r, "stake") for r in open_rows), me.ZERO),
            "settled": len(closed),
            "wins": len(wins),
            "win_rate": (Decimal(len(wins)) / len(closed)) if closed else None,
            "pnl": pnl,
            "staked": staked,
            "roi": (pnl / staked) if staked > 0 else None,
        }


class RiskManager:
    """
    Non-negotiable circuit breakers, re-checked immediately before any trade is executed:
      * daily drawdown stop (sticky for the rest of the day once tripped)
      * duplicate position / resting-order prevention (per ticker and per exposure group)
      * max positions per game (ML + spread + total + props on one game are correlated)
      * single-bet cap (min of % of equity and hard dollar cap)
      * total exposure cap and available-cash check

    Day-start equity is the first equity observed each day (DAY_TZ); deposits or
    withdrawals during the day will distort the drawdown figure.
    """

    def __init__(self, state_path: Path | None = None, now_fn: Callable[[], datetime] | None = None):
        self.state_path = Path(state_path or config.RISK_STATE_PATH)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.last_status: dict[str, Any] = {}

    def _load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def reset_source(self, source: str) -> None:
        """Forget the day-start equity / halt state of one account source (e.g. 'paper') - used by Reset Paper Balance."""
        state = self._load()
        self._save({k: v for k, v in state.items() if not k.startswith(f"{source}:")})

    def check_halt(self, account: AccountState) -> bool:
        """Update the day's drawdown from current equity; True means trading is halted."""
        state = self._load()
        key = f"{account.source}:{config.PAPER_DATA_ENV if account.source == 'paper' else config.KALSHI_ENV}"
        today = self._now().astimezone(config.DAY_TZ).date().isoformat()
        entry = state.get(key)
        if not entry or entry.get("date") != today:
            entry = {"date": today, "start_equity": str(account.equity), "halted": False, "halt_reason": ""}

        start = Decimal(entry["start_equity"])
        drawdown = (start - account.equity) / start if start > 0 else me.ZERO
        if not entry["halted"] and drawdown >= config.MAX_DAILY_DRAWDOWN:
            entry["halted"] = True
            entry["halt_reason"] = (
                f"daily drawdown {drawdown:.2%} >= limit {config.MAX_DAILY_DRAWDOWN:.2%} "
                f"(start {start:.2f}, now {account.equity:.2f})"
            )
        state[key] = entry
        self._save(state)
        self.last_status = {
            "date": today, "start_equity": start, "equity": account.equity,
            "drawdown": drawdown, "halted": entry["halted"], "reason": entry["halt_reason"],
        }
        return bool(entry["halted"])

    def pre_trade_check(self, plan: "OrderPlan", account: AccountState) -> tuple[Decision | None, str]:
        """Final gate. Returns (None, 'ok') when the order may proceed."""
        if self.check_halt(account):
            return Decision.HALT_DRAWDOWN, self.last_status.get("reason", "trading halted")
        if account.has_exposure(plan.ticker, plan.group):
            return Decision.SKIP_DUPLICATE, "open position or resting order already exists for this game/player/ticker"
        if account.game_positions(game_code_of(plan.ticker)) >= config.MAX_POSITIONS_PER_GAME:
            return Decision.SKIP_RISK, f"already {config.MAX_POSITIONS_PER_GAME} positions in this game"
        max_bet = min(config.MAX_BET_FRACTION * account.equity, config.MAX_BET_DOLLARS)
        if plan.total_cost > max_bet:
            return Decision.SKIP_RISK, f"stake {plan.total_cost:.2f} exceeds single-bet cap {max_bet:.2f}"
        if plan.total_cost > account.cash:
            return Decision.SKIP_RISK, f"stake {plan.total_cost:.2f} exceeds available cash {account.cash:.2f}"
        exposure_cap = account.equity * config.MAX_TOTAL_EXPOSURE_FRACTION
        if account.exposure + plan.total_cost > exposure_cap:
            return Decision.SKIP_RISK, f"total exposure would exceed cap {exposure_cap:.2f}"
        return None, "ok"


# =============================================================================
# Orders and evaluation
# =============================================================================
@dataclass
class OrderPlan:
    ticker: str
    event_ticker: str
    side: str  # "yes" | "no"
    price: Decimal
    contracts: int
    route: str  # "maker" (resting, post-only) | "taker" (fill-or-kill at the ask)
    fee_total: Decimal
    total_cost: Decimal  # price * contracts + fee = maximum loss
    net_edge: Decimal
    fair_side: Decimal  # devigged consensus probability of this side winning
    market_type: str = "moneyline"
    label: str = ""
    min_edge: Decimal = config.MIN_NET_EDGE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def post_only(self) -> bool:
        return self.route == "maker"

    @property
    def fill_or_kill(self) -> bool:
        return self.route == "taker"

    @property
    def group(self) -> str:
        return event_from_ticker(self.ticker)


@dataclass
class Evaluation:
    decision: Decision
    reason: str
    rec: dict[str, Any]
    plan: OrderPlan | None = None


@dataclass
class Pick:
    """A qualifying, not-yet-executed order proposal produced by a scan."""

    plan: OrderPlan
    rec: dict[str, Any]
    reason: str
    run_id: str


class ExecutionEngine:
    """Evaluates every matched market and PROPOSES picks. It never sends or records a trade itself."""

    def __init__(
        self,
        kalshi: KalshiClient,
        journal: TradeJournal,
        risk: RiskManager,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.kalshi = kalshi
        self.journal = journal
        self.risk = risk
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.counts: Counter[str] = Counter()
        self.picks: list[Pick] = []
        self.book_calls = 0

    @staticmethod
    def min_edge(category: str) -> Decimal:
        return config.MIN_NET_EDGE + config.CATEGORIES[category].edge_buffer

    # ------------------------------------------------------------------ logging helpers
    def _log(self, decision: Decision, reason: str, rec: dict[str, Any]) -> None:
        self.counts[decision.value] += 1
        self.journal.log(decision, reason, rec)

    def log_unmatched_odds(self, event: OddsEvent, snapshot: OddsSnapshot) -> None:
        now = self._now()
        self._log(Decision.SKIP_NO_MATCH, "no Kalshi NBA market found for this Odds API game", {
            "event_id": event.event_id,
            "team_yes": f"{event.away_team} @ {event.home_team}",
            "commence_time": event.commence_time,
            "minutes_to_tipoff": (event.commence_time - now).total_seconds() / 60,
            "odds_age_seconds": snapshot.age_seconds(now),
        })

    def log_unmatched_kalshi(self, game_label: str) -> None:
        self._log(Decision.SKIP_NO_MATCH, "Kalshi game has no matching Odds API game", {"team_yes": game_label})

    def log_unresolved(self, unresolved: list[tuple[dict, str]]) -> None:
        """Kalshi markets whose teams/player/line could not be read unambiguously (one row per distinct reason)."""
        by_reason: dict[str, list[dict]] = defaultdict(list)
        for raw, reason in unresolved:
            by_reason[reason].append(raw)
        for reason, raws in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))[:20]:
            sample = raws[0]
            self._log(Decision.SKIP_NO_MATCH, f"{len(raws)} Kalshi market(s) not parseable: {reason}", {
                "kalshi_ticker": sample.get("ticker", ""),
                "team_yes": sample.get("yes_sub_title") or sample.get("title") or "",
                "market_type": "unparsed",
            })

    # ------------------------------------------------------------------ per-game pipeline
    def process_game(self, game: MatchedGame, snapshot: OddsSnapshot, account: AccountState) -> None:
        now = self._now()
        event = game.odds_event
        minutes_to_tipoff = Decimal(str((event.commence_time - now).total_seconds())) / 60
        base: dict[str, Any] = {
            "event_id": event.event_id,
            "commence_time": event.commence_time,
            "minutes_to_tipoff": round(minutes_to_tipoff, 1),
            "odds_age_seconds": round(Decimal(str(snapshot.age_seconds(now))), 1),
            "bankroll": account.equity,
            "cash": account.cash,
        }
        by_category: dict[str, list[MarketSpec]] = defaultdict(list)
        for spec in game.specs:
            by_category[spec.category].append(spec)

        def log_categories(decision: Decision, reason: str, extra: dict | None = None) -> None:
            for category in config.CATEGORIES:
                if category in by_category:
                    self._log(decision, reason, {**base, **(extra or {}), "market_type": category,
                                                 "team_yes": game.label,
                                                 "market_label": f"{len(by_category[category])} Kalshi market(s)"})

        # --- global gates (no API calls) -------------------------------------------------
        if self.risk.check_halt(account):
            log_categories(Decision.HALT_DRAWDOWN, self.risk.last_status.get("reason", "trading halted"))
            return
        if config.SKIP_LIVE_GAMES and minutes_to_tipoff < config.MIN_MINUTES_BEFORE_TIPOFF:
            log_categories(Decision.SKIP_LIVE, f"game starts in {minutes_to_tipoff:.1f} min (or already started)")
            return
        if snapshot.age_seconds(now) > config.MAX_ODDS_AGE_SECONDS:
            log_categories(Decision.SKIP_STALE, f"odds snapshot is {snapshot.age_seconds(now):.0f}s old "
                                                f"(> {config.MAX_ODDS_AGE_SECONDS}s)")
            return

        fresh, stale = event.fresh_quotes(now, config.MAX_BOOK_UPDATE_AGE_SECONDS)
        if not fresh:
            if stale:
                oldest = max((now - q.last_update).total_seconds() for q in stale)
                log_categories(Decision.SKIP_STALE, f"all {len(stale)} book quotes older than "
                                                    f"{config.MAX_BOOK_UPDATE_AGE_SECONDS}s (oldest {oldest:.0f}s)")
            else:
                log_categories(Decision.SKIP_NO_ODDS, "no bookmaker quotes for this game")
            return
        base["stale_quotes_dropped"] = len(stale)

        # --- index fresh quotes by exact selection: SelectionKey -> {book_key: american price}
        index: dict[Any, dict[str, Decimal]] = defaultdict(dict)
        players: dict[str, set[str]] = defaultdict(set)
        for quote in fresh:
            for key, price in quote.selections.items():
                index[key].setdefault(quote.book_key, price)
                if key.market.startswith("player_"):
                    players[key.market].add(key.subject)

        # --- per-market evaluation (matching, consensus, prescreen, order book) ------------
        evaluations: list[tuple[MarketSpec, Evaluation]] = []
        no_line: Counter[str] = Counter()
        for category in config.CATEGORIES:
            specs = by_category.get(category)
            if not specs:
                continue
            rules = config.CATEGORIES[category]
            for spec in specs:
                evaluation = self._evaluate_spec(spec, rules, index, players, account, base)
                if evaluation is None:
                    no_line[category] += 1
                else:
                    evaluations.append((spec, evaluation))
        for category, count in no_line.items():
            self._log(Decision.SKIP_NO_MATCH,
                      f"{count} Kalshi {config.CATEGORIES[category].label.lower()} market(s) had no sportsbook "
                      f"quote on the exact same line/player (not comparable)",
                      {**base, "market_type": category, "team_yes": game.label})

        # --- best candidate per exposure group, then propose in edge order ------------------
        best: dict[str, tuple[MarketSpec, Evaluation]] = {}
        for spec, evaluation in evaluations:
            if evaluation.plan is None:
                continue
            current = best.get(spec.group)
            if current is None or (evaluation.plan.net_edge, evaluation.plan.total_cost) > (
                    current[1].plan.net_edge, current[1].plan.total_cost):
                best[spec.group] = (spec, evaluation)
        chosen = {id(e) for _, e in best.values()}
        for _spec, evaluation in evaluations:
            if id(evaluation) in chosen:
                continue
            if evaluation.plan is not None:
                self._log(Decision.SKIP_DUPLICATE,
                          "same-game/same-player candidate with a higher net edge was selected", evaluation.rec)
            else:
                self._log(evaluation.decision, evaluation.reason, evaluation.rec)
        for _spec, evaluation in sorted(best.values(), key=lambda t: t[1].plan.net_edge, reverse=True):
            self._propose(evaluation, account)

    # ------------------------------------------------------------------ market evaluation
    def _evaluate_spec(
        self, spec: MarketSpec, rules: config.MarketCategory, index: dict, players: dict[str, set[str]],
        account: AccountState, base: dict[str, Any],
    ) -> Evaluation | None:
        """Returns None when no sportsbook quotes this exact selection (the caller summarises those)."""
        yes_key, no_key = spec.yes_key, spec.no_key
        if spec.category == "prop" and yes_key not in index:  # tolerate "Nic" vs "Nicolas" style differences
            candidate = me.resolve_player(yes_key.subject, players.get(spec.odds_market, ()))
            if candidate and candidate != yes_key.subject:
                yes_key, no_key = yes_key._replace(subject=candidate), no_key._replace(subject=candidate)
        yes_books, no_books = index.get(yes_key), index.get(no_key)
        if not yes_books or not no_books:
            return None
        common = [book for book in yes_books if book in no_books]
        if not common:
            return None

        min_edge = self.min_edge(spec.category)
        rec: dict[str, Any] = {
            **base, "market_type": spec.category, "market_label": spec.title, "line": spec.line,
            "kalshi_event_ticker": spec.event_ticker, "kalshi_ticker": spec.ticker, "team_yes": spec.yes_label,
        }

        def done(decision: Decision, reason: str, plan: OrderPlan | None = None) -> Evaluation:
            return Evaluation(decision, reason, rec, plan)

        consensus = me.build_consensus([(b, {"yes": yes_books[b], "no": no_books[b]}) for b in common], ("yes", "no"))
        if consensus is None:
            return done(Decision.SKIP_NO_ODDS, "no usable two-way quotes on this exact line")
        fair_yes = consensus.probs["yes"]
        rec.update({
            "fair_prob_yes_team": round(fair_yes, 5),
            "books_used": consensus.books_used,
            "sharp_books": consensus.sharp_books_used,
            "sharp_weight_share": round(consensus.sharp_weight_share, 4),
            "consensus_dispersion": round(consensus.dispersion, 5),
        })
        if len(consensus.books_used) < rules.min_books:
            return done(Decision.SKIP_CONSENSUS, f"only {len(consensus.books_used)} fresh book(s) quote this exact "
                                                 f"line; need {rules.min_books}")
        if rules.require_sharp and not consensus.sharp_books_used:
            return done(Decision.SKIP_CONSENSUS, "no fresh sharp book quotes this exact line")
        if account.has_exposure(spec.ticker, spec.group):
            return done(Decision.SKIP_DUPLICATE, "open position or resting order already exists here")

        try:
            fee_schedule = self.kalshi.get_fee_schedule(spec.series)
        except Exception as exc:  # noqa: BLE001
            return done(Decision.ERROR, f"fee schedule lookup failed: {exc}")

        # --- prescreen with the quotes already in the /markets listing (no extra API call) ------
        if spec.yes_bid is not None and spec.no_bid is not None and spec.yes_bid + spec.no_bid < me.ONE:
            quick = max(me.price_sides(fair_yes, spec.yes_bid, spec.no_bid, fee_schedule), key=lambda s: s["maker_edge"])
            if quick["maker_edge"] < min_edge - config.PRESCREEN_SLACK:
                rec.update(side=quick["side"], fair_prob_side=round(quick["fair"], 5), net_edge=round(quick["maker_edge"], 5),
                           yes_bid=spec.yes_bid, no_bid=spec.no_bid)
                return done(Decision.PASS, f"prescreen: best edge {quick['maker_edge']:.4f} from listing quotes is "
                                           f"far below min {min_edge:.4f}")

        # --- order book --------------------------------------------------------------------
        if self.book_calls >= config.MAX_ORDERBOOK_CALLS_PER_SCAN:
            return done(Decision.SKIP_BUDGET, f"order-book call budget ({config.MAX_ORDERBOOK_CALLS_PER_SCAN}) "
                                              f"for this scan is used up")
        self.book_calls += 1
        try:
            book = self.kalshi.get_orderbook(spec.ticker)
        except Exception as exc:  # noqa: BLE001 - journal every failure, never crash the scan
            rec["api_status"] = "orderbook_error"
            rec["api_response"] = str(exc)
            return done(Decision.ERROR, f"order book fetch failed: {exc}")

        yes_bid, no_bid = book.best_yes_bid, book.best_no_bid
        rec.update(yes_bid=yes_bid, no_bid=no_bid, yes_ask=book.best_ask("yes"),
                   no_ask=book.best_ask("no"), spread=book.spread)
        if yes_bid is None or no_bid is None:
            return done(Decision.SKIP_NO_BOOK, "order book is empty or one-sided")
        spread = book.spread
        if spread <= 0:
            return done(Decision.SKIP_NO_BOOK, f"locked/crossed book (spread {spread})")
        if spread > rules.max_spread:
            return done(Decision.SKIP_SPREAD, f"spread {spread:.2f} > max {rules.max_spread:.2f}")

        sides = me.price_sides(fair_yes, yes_bid, no_bid, fee_schedule)
        rec["details"] = {
            "edge_by_side": {s["side"]: {"maker_price": s["maker_price"], "maker_edge": round(s["maker_edge"], 5),
                                         "taker_edge": round(s["taker_edge"], 5)} for s in sides},
            "fee_type": fee_schedule.fee_type, "fee_multiplier": fee_schedule.multiplier,
        }
        best = max(sides, key=lambda s: s["maker_edge"])
        side, fair_side, ask = best["side"], best["fair"], best["ask"]
        rec.update(side=side, fair_prob_side=round(fair_side, 5), order_price=best["maker_price"],
                   net_edge=round(best["maker_edge"], 5),
                   pick_label=spec.yes_label if side == "yes" else spec.no_label)
        if best["maker_edge"] < min_edge:
            return done(Decision.PASS, f"best net edge {best['maker_edge']:.4f} < min {min_edge:.4f}")

        depth = book.ask_depth(side)
        rec["ask_depth"] = depth
        if depth < config.MIN_ASK_DEPTH_CONTRACTS:
            return done(Decision.SKIP_LIQUIDITY,
                        f"only {depth} contracts at the best ask (need >= {config.MIN_ASK_DEPTH_CONTRACTS})")

        # --- routes in priority order: maker-first, taker only for a large, fee-adjusted edge
        routes: list[tuple[str, Decimal]] = []
        if config.ALLOW_TAKER_ORDERS and best["taker_edge"] >= max(config.TAKER_OVERRIDE_MIN_EDGE, min_edge):
            routes.append(("taker", ask))
        routes.append(("maker", best["maker_price"]))

        room = max(account.equity * config.MAX_TOTAL_EXPOSURE_FRACTION - account.exposure, me.ZERO)
        last_reason, last_decision = "no route available", Decision.PASS
        for route, price in routes:
            sizing = me.size_order(fair_prob=fair_side, price=price, role=route, schedule=fee_schedule,
                                   equity=account.equity, cash_available=account.cash, exposure_room=room)
            rec.update(route=route, order_price=price, kelly_full=round(sizing.kelly_full, 5),
                       kelly_applied=round(sizing.kelly_applied, 5), binding_cap=sizing.binding_cap)
            if sizing.contracts < 1:
                last_decision = Decision.SKIP_RISK
                last_reason = (f"{route}: position rounds to 0 contracts (budget {sizing.budget:.2f}, "
                               f"binding cap '{sizing.binding_cap}')")
                continue
            fee_each = me.fee_per_contract_exact(price, sizing.contracts, route, fee_schedule)
            exact_edge = me.net_edge(fair_side, price, fee_each)
            rec.update(contracts=sizing.contracts, stake=sizing.total_cost, fee_total=sizing.fee,
                       fee_per_contract=round(fee_each, 5), net_edge=round(exact_edge, 5))
            if exact_edge < min_edge:
                last_decision = Decision.PASS
                last_reason = (f"{route}: net edge {exact_edge:.4f} at {sizing.contracts} contracts "
                               f"(after exact fee rounding) < min {min_edge:.4f}")
                continue
            if route == "taker" and depth < sizing.contracts * config.DEPTH_COVERAGE_MULTIPLE:
                last_decision = Decision.SKIP_LIQUIDITY
                last_reason = f"taker: {depth} contracts at the ask < {sizing.contracts} needed"
                continue
            plan = OrderPlan(
                ticker=spec.ticker, event_ticker=spec.event_ticker, side=side, price=price,
                contracts=sizing.contracts, route=route, fee_total=sizing.fee,
                total_cost=sizing.total_cost, net_edge=exact_edge, fair_side=fair_side,
                market_type=spec.category, label=rec["pick_label"], min_edge=min_edge,
                created_at=self._now())
            rec["client_order_id"] = plan.client_order_id
            return done(Decision.BUY, f"{route} order, net edge {exact_edge:.4f}", plan)
        return done(last_decision, last_reason)

    # ------------------------------------------------------------------ proposals
    def _propose(self, evaluation: Evaluation, account: AccountState) -> None:
        plan, rec = evaluation.plan, evaluation.rec
        assert plan is not None
        decision, why = self.risk.pre_trade_check(plan, account)
        if decision is not None:
            self._log(decision, why, rec)
            return
        account.register(plan)  # later games in this scan size against the exposure already proposed
        self.picks.append(Pick(plan=plan, rec=rec, reason=evaluation.reason, run_id=self.journal.run_id))
        self._log(Decision.BUY, evaluation.reason, rec)

    # ------------------------------------------------------------------ scan driver
    def process_all(self, games: list[MatchedGame], snapshot: OddsSnapshot, account: AccountState) -> None:
        # soonest tip-off first: they are the most time-sensitive and get the order-book budget first
        for game in sorted(games, key=lambda g: g.odds_event.commence_time):
            try:
                self.process_game(game, snapshot, account)
            except Exception as exc:  # noqa: BLE001 - one bad game must not kill the scan
                log.exception("Unhandled error processing %s", game.game_code)
                self._log(Decision.ERROR, f"unhandled error: {exc}",
                          {"kalshi_event_ticker": game.game_code, "event_id": game.odds_event.event_id,
                           "team_yes": game.label})


# =============================================================================
# EXECUTION: paper / demo
# =============================================================================
def new_trade_row(
    plan: OrderPlan, *, mode: str, status: str, env: str, filled=None, stake: Decimal | None = None,
    order_id: str = "", note: str = "", now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    return {
        "timestamp": now.isoformat(timespec="seconds"),
        "ticker": plan.ticker,
        "market_type": plan.market_type,
        "side": plan.side.upper(),
        "price": plan.price,
        "contracts": plan.contracts,
        "devigged_prob": f"{plan.fair_side:.5f}",
        "edge_pct": f"{plan.net_edge * 100:.2f}",
        "status": status,
        "trade_id": uuid.uuid4().hex[:12],
        "mode": mode,
        "env": env,
        "label": plan.label,
        "event_ticker": plan.event_ticker,
        "route": plan.route,
        "stake": plan.total_cost if stake is None else stake,
        "fee_total": plan.fee_total,
        "filled": plan.contracts if filled is None else filled,
        "order_id": order_id,
        "client_order_id": plan.client_order_id,
        "note": note,
    }


def execute_paper(
    pick: Pick, *, ledger: TradeLedger, journal: TradeJournal, risk: RiskManager,
    mode: str = "paper", now: datetime | None = None,
) -> tuple[bool, str]:
    """
    Simulated fill. Re-checks every risk rule against the ledger-derived paper account, then appends the
    trade to the ledger as OPEN (stake = price x contracts + fee, deducted from paper cash).
    NOTE: a paper maker order is assumed to fill at its limit price; real resting orders may not.
    """
    now = now or datetime.now(timezone.utc)
    account = ledger.account(now, mode)
    decision, why = risk.pre_trade_check(pick.plan, account)
    if decision is not None:
        journal.log(decision, f"paper trade blocked: {why}", pick.rec)
        return False, why
    ledger.append(new_trade_row(pick.plan, mode=mode, status="OPEN", env="simulated", now=now,
                                note="fill assumed at limit price"))
    return True, "Paper trade recorded."


# =============================================================================
# EXECUTION: real money
# =============================================================================
@dataclass
class LiveResult:
    ok: bool
    message: str
    status: str = ""
    order_id: str = ""
    plan: OrderPlan | None = None


def reprice_plan(plan: OrderPlan, kalshi: KalshiClient) -> tuple[OrderPlan | None, str]:
    """Re-price a pick against a FRESH order book; refuse if the edge is gone or the book moved too far."""
    try:
        book = kalshi.get_orderbook(plan.ticker)
        schedule = kalshi.get_fee_schedule(plan.ticker.split("-")[0])
    except Exception as exc:  # noqa: BLE001
        return None, f"could not refresh the order book: {exc}"
    bid, ask = book.best_bid(plan.side), book.best_ask(plan.side)
    if bid is None or ask is None or (book.spread is not None and book.spread <= 0):
        return None, "order book is now empty, one-sided or crossed"
    if plan.route == "taker":
        price = ask
        edge = me.net_edge(plan.fair_side, price, me.fee_per_contract_exact(price, plan.contracts, "taker", schedule))
        if edge < max(config.TAKER_OVERRIDE_MIN_EDGE, plan.min_edge):
            return None, f"price moved: ask is now ${price:.2f}, net edge {edge:.4f} is below the taker threshold"
        if book.ask_depth(plan.side) < plan.contracts * config.DEPTH_COVERAGE_MULTIPLE:
            return None, "not enough contracts at the ask any more"
    else:
        price = min(bid + config.MAKER_PRICE_IMPROVEMENT, ask - config.TICK)
        price = min(max(price, config.TICK), me.ONE - config.TICK)
        edge = me.net_edge(plan.fair_side, price, me.fee_per_contract_exact(price, plan.contracts, "maker", schedule))
        if edge < plan.min_edge:
            return None, f"price moved: net edge is now {edge:.4f}, below the {plan.min_edge:.4f} minimum"
    fee = me.fee_total(price, plan.contracts, plan.route, schedule)
    return replace(plan, price=price, fee_total=fee, total_cost=price * plan.contracts + fee, net_edge=edge), "ok"


def place_live_order(
    pick: Pick, *, kalshi: KalshiClient, journal: TradeJournal, risk: RiskManager, ledger: TradeLedger,
    confirmed: bool, now: datetime | None = None,
) -> LiveResult:
    """
    Send ONE real order. Nothing is transmitted unless `confirmed` is True (the app passes True only
    after the "Enable Live Real Money" box is ticked AND the user presses this trade's confirm button).

    Steps: age check -> live account snapshot -> re-price on a fresh book -> full risk check on the live
    account -> RSA-PSS signed POST /portfolio/orders -> ledger row (status from the order response).
    """
    if not confirmed:
        return LiveResult(False, "Not confirmed - no order was sent.")
    now = now or datetime.now(timezone.utc)
    plan = pick.plan
    if not getattr(kalshi, "has_credentials", False):
        return LiveResult(False, "Kalshi API credentials are not loaded - no order was sent.")
    age = (now - plan.created_at).total_seconds()
    if age > config.LIVE_PLAN_MAX_AGE_SECONDS:
        return LiveResult(False, f"This pick is {age:.0f}s old (limit {config.LIVE_PLAN_MAX_AGE_SECONDS}s). Rescan first.")

    try:
        account = build_live_account(kalshi)
    except Exception as exc:  # noqa: BLE001
        return LiveResult(False, f"Could not read your Kalshi account: {exc}")
    fresh, why = reprice_plan(plan, kalshi)
    if fresh is None:
        journal.log(Decision.SKIP_RISK, f"live order refused at confirm time: {why}", pick.rec)
        return LiveResult(False, f"Order not sent: {why}")
    decision, why = risk.pre_trade_check(fresh, account)
    if decision is not None:
        journal.log(decision, f"live order refused at confirm time: {why}", pick.rec)
        return LiveResult(False, f"Order not sent: {why}")

    env = config.KALSHI_ENV
    rec = {**pick.rec, "order_price": fresh.price, "contracts": fresh.contracts, "stake": fresh.total_cost,
           "net_edge": round(fresh.net_edge, 5), "route": fresh.route}
    try:
        response = kalshi.place_order(
            ticker=fresh.ticker, side=fresh.side, count=fresh.contracts, price=fresh.price,
            client_order_id=fresh.client_order_id, post_only=fresh.post_only, fill_or_kill=fresh.fill_or_kill)
    except Exception as exc:  # noqa: BLE001
        rec.update(api_status="order_error", api_response=str(exc))
        journal.log(Decision.ERROR, f"live order failed: {exc}", rec)
        ledger.append(new_trade_row(fresh, mode="live", status="REJECTED", env=env, filled=0, stake=me.ZERO,
                                    note=str(exc)[:300], now=now))
        return LiveResult(False, f"Kalshi rejected the order: {exc}", plan=fresh)

    order = response.get("order") if isinstance(response, dict) else None
    summary = summarize_order(order) if isinstance(order, dict) else None
    order_id = (order or {}).get("order_id", "") if isinstance(order, dict) else ""
    note = "duplicate client_order_id: order already accepted" if (isinstance(response, dict) and response.get("duplicate_client_order_id")) else ""
    if summary is None:
        status, filled, stake = "RESTING", me.ZERO, fresh.total_cost
    elif summary.filled == 0 and summary.status == "canceled":
        status, filled, stake = "CANCELED", me.ZERO, me.ZERO
        note = note or "fill-or-kill order found no liquidity and was canceled (nothing charged)"
    elif summary.status == "executed" or summary.filled >= fresh.contracts or (summary.filled > 0 and summary.status == "canceled"):
        status, filled = "OPEN", summary.filled
        stake = summary.cost + summary.fees if summary.cost > 0 else fresh.total_cost
    else:
        status, filled, stake = "RESTING", summary.filled, fresh.total_cost
    rec.update(api_status="accepted", api_response=response, order_id=order_id)
    journal.log(Decision.BUY, f"LIVE order sent ({fresh.route}); Kalshi status {summary.status if summary else 'unknown'}", rec)
    ledger.append(new_trade_row(fresh, mode="live", status=status, env=env, filled=filled, stake=stake,
                                order_id=order_id, note=note, now=now))
    messages = {
        "OPEN": f"Filled {filled} contract(s). Order {order_id}.",
        "RESTING": f"Resting limit order placed. It will fill only if the market trades at ${fresh.price:.2f}. Order {order_id}.",
        "CANCELED": "Order canceled: no liquidity at that price (nothing charged).",
    }
    return LiveResult(status != "CANCELED", messages.get(status, status), status=status, order_id=order_id, plan=fresh)


# =============================================================================
# SETTLEMENT
# =============================================================================
@dataclass
class SettleReport:
    checked: int = 0
    settled: int = 0
    updated: int = 0
    canceled: int = 0
    errors: list[str] = field(default_factory=list)


def settle_trades(
    ledger: TradeLedger, kalshi, *, modes: Iterable[str] = ("paper", "live"), simulate: bool = False,
    now: datetime | None = None,
) -> SettleReport:
    """
    Check every OPEN / RESTING trade against Kalshi and book the result.

      paper   market settled -> payout = $1 per contract on the winning side; P&L = payout - stake
      live    first refreshes the order (fills, fees, canceled?) via GET /portfolio/orders/{id}, then settles
      void    market ended without a yes/no result -> stake refunded, P&L 0
      simulate=True (Offline Demo only) draws each result from the trade's own devigged probability
    """
    now = now or datetime.now(timezone.utc)
    report = SettleReport()
    updates: dict[str, dict] = {}
    can_auth = bool(getattr(kalshi, "has_credentials", False))
    for row in ledger.rows():
        if row.get("status") not in OPEN_STATUSES or row.get("mode") not in tuple(modes):
            continue
        report.checked += 1
        trade_id, live = row["trade_id"], row.get("mode") == "live"
        try:
            fields: dict[str, Any] = {}
            filled, stake = _dec(row, "filled", _dec(row, "contracts")), _dec(row, "stake")
            if live and not simulate:
                if not can_auth:
                    report.errors.append(f"{row['ticker']}: real trades need Kalshi credentials to refresh")
                    continue
                summary = summarize_order(kalshi.get_order(row["order_id"]))
                filled = summary.filled
                if summary.status in ("canceled", "executed") and filled == 0:
                    updates[trade_id] = {"status": "CANCELED", "filled": 0, "stake": 0, "payout": 0, "pnl": 0,
                                         "settled_at": now.isoformat(timespec="seconds"),
                                         "note": "order canceled with no fills"}
                    report.canceled += 1
                    continue
                if filled > 0 and summary.cost > 0:
                    stake = summary.cost + summary.fees
                fields.update(filled=filled, stake=stake)
                fields["status"] = "OPEN" if filled > 0 else "RESTING"

            if simulate:
                rng = random.Random(trade_id)
                won = Decimal(str(rng.random())) < _dec(row, "devigged_prob", Decimal("0.5"))
                side = row["side"].lower()
                result = side if won else ("no" if side == "yes" else "yes")
            else:
                result = settlement_result(kalshi.get_market(row["ticker"]))

            if result is None:
                if fields:
                    updates[trade_id] = fields
                    report.updated += 1
                continue
            if result == "void":
                payout, status, pnl = stake, "VOID", me.ZERO
            else:
                payout = me.settle_payout(row["side"].lower(), result, filled)
                status, pnl = ("WON" if payout > 0 else "LOST"), payout - stake
            fields.update(status=status, result=result.upper(), payout=payout, pnl=pnl,
                          settled_at=now.isoformat(timespec="seconds"), filled=filled, stake=stake)
            updates[trade_id] = fields
            report.settled += 1
        except Exception as exc:  # noqa: BLE001 - one bad trade must not stop the others
            report.errors.append(f"{row.get('ticker', '?')}: {exc}")
    ledger.update_many(updates)
    return report
