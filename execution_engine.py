"""
execution_engine.py - Matching, gating, sizing, risk control, journaling and order placement.

Per-game decision pipeline (cheap gates first, API calls last):

    HALT_DRAWDOWN -> SKIP_LIVE -> SKIP_STALE -> SKIP_NO_ODDS / SKIP_CONSENSUS
      -> SKIP_DUPLICATE -> [per market: order book -> SKIP_NO_BOOK -> SKIP_SPREAD -> PASS
         -> SKIP_LIQUIDITY -> SKIP_RISK -> BUY]

Only the single best candidate per game can become a BUY; the engine never holds both
sides of one game. Every evaluated market is written to the journal (CSV + JSONL).

Safety: real orders are sent only when config.DRY_RUN is False AND the KALSHI_ARM_LIVE
environment variable matches config.LIVE_ARM_PHRASE. Otherwise orders are simulated and
recorded in a paper ledger so duplicate / exposure checks still work across scans.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import config
import math_engine as me
from kalshi_client import KalshiClient
from odds_client import OddsEvent, OddsSnapshot

log = logging.getLogger("execution_engine")


# =============================================================================
# Decisions
# =============================================================================
class Decision(str, Enum):
    BUY = "BUY"
    PASS = "PASS"  # priced, but net edge below threshold
    SKIP_SPREAD = "SKIP_SPREAD"
    SKIP_LIQUIDITY = "SKIP_LIQUIDITY"
    SKIP_STALE = "SKIP_STALE"
    SKIP_NO_ODDS = "SKIP_NO_ODDS"
    SKIP_CONSENSUS = "SKIP_CONSENSUS"  # too few books / no sharp book
    SKIP_NO_BOOK = "SKIP_NO_BOOK"  # empty, one-sided, locked or crossed order book
    SKIP_NO_MATCH = "SKIP_NO_MATCH"  # no Kalshi<->Odds API pairing
    SKIP_LIVE = "SKIP_LIVE"
    SKIP_DUPLICATE = "SKIP_DUPLICATE"
    SKIP_RISK = "SKIP_RISK"
    HALT_DRAWDOWN = "HALT_DRAWDOWN"
    ERROR = "ERROR"


# =============================================================================
# Audit journal (CSV + JSONL)
# =============================================================================
JOURNAL_FIELDS = [
    "ts_utc", "run_id", "decision", "reason", "route", "mode",
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


class TradeJournal:
    """Append-only audit log. CSV is for spreadsheets; JSONL keeps the full API responses."""

    def __init__(self, csv_path: Path | None = None, jsonl_path: Path | None = None, run_id: str | None = None):
        self.csv_path = Path(csv_path or config.JOURNAL_CSV_PATH)
        self.jsonl_path = Path(jsonl_path or config.JOURNAL_JSONL_PATH)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(JOURNAL_FIELDS)

    def log(self, decision: Decision, reason: str, rec: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "decision": decision.value,
            "reason": reason,
            "mode": "DRY_RUN" if config.DRY_RUN else "LIVE",
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
# Kalshi <-> Odds API matching
# =============================================================================
_EVENT_RE = re.compile(r"^(?P<series>[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<teams>[A-Z]{4,8})$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def canonical_code(code: str) -> str | None:
    code = code.upper()
    code = config.NBA_TEAM_CODE_ALIASES.get(code, code)
    return code if code in config.NBA_TEAMS else None


def parse_event_ticker(event_ticker: str) -> tuple[date, str, str] | None:
    """'KXNBAGAME-26NOV17LALHOU' -> (2026-11-17, 'LAL', 'HOU') (Kalshi lists away team first)."""
    match = _EVENT_RE.match(event_ticker.upper())
    if not match:
        return None
    month = _MONTHS.get(match["mon"])
    if not month:
        return None
    try:
        game_date = date(2000 + int(match["yy"]), month, int(match["dd"]))
    except ValueError:
        return None
    teams = match["teams"]
    for split in range(2, 5):
        first, second = canonical_code(teams[:split]), canonical_code(teams[split:])
        if first and second and first != second:
            return game_date, first, second
    return None


def odds_name_to_code(team_name: str) -> str | None:
    lowered = team_name.lower()
    for code, (_city, nickname) in config.NBA_TEAMS.items():
        if nickname.lower() in lowered:
            return code
    return None


def event_from_ticker(market_ticker: str) -> str:
    return market_ticker.rsplit("-", 1)[0]


def _market_yes_team_code(market: dict, team_codes: tuple[str, str]) -> str | None:
    suffix = canonical_code(market["ticker"].rsplit("-", 1)[-1])
    if suffix in team_codes:
        return suffix
    subtitle = (market.get("yes_sub_title") or "").lower()
    for index in (1, 0):  # nickname first, then city
        hits = [c for c in team_codes if config.NBA_TEAMS[c][index].lower() in subtitle]
        if len(hits) == 1:
            return hits[0]
    return None


@dataclass
class MatchedGame:
    odds_event: OddsEvent
    kalshi_event_ticker: str
    series_ticker: str
    markets: list[dict]  # each carries "_yes_team" = the Odds API team name its YES side wins on
    unresolved: list[dict] = field(default_factory=list)


def match_games(
    odds_events: list[OddsEvent], kalshi_markets: list[dict]
) -> tuple[list[MatchedGame], list[OddsEvent], list[str]]:
    """Pair Odds API games with Kalshi events on (team pair, ET game date +-1 day)."""
    by_event: dict[str, list[dict]] = {}
    for market in kalshi_markets:
        ticker = market.get("ticker", "")
        if not ticker:
            continue
        by_event.setdefault(market.get("event_ticker") or event_from_ticker(ticker), []).append(market)

    parsed: dict[str, tuple[date, str, str]] = {}
    unmatched_kalshi: list[str] = []
    for event_ticker in by_event:
        info = parse_event_ticker(event_ticker)
        if info:
            parsed[event_ticker] = info
        else:
            unmatched_kalshi.append(event_ticker)

    games: list[MatchedGame] = []
    unmatched_odds: list[OddsEvent] = []
    used: set[str] = set()
    for event in odds_events:
        home_code, away_code = odds_name_to_code(event.home_team), odds_name_to_code(event.away_team)
        if not home_code or not away_code:
            unmatched_odds.append(event)
            continue
        pair = frozenset({home_code, away_code})
        et_date = event.commence_time.astimezone(config.GAME_TZ).date()
        best: tuple[int, str] | None = None
        for event_ticker, (game_date, code_a, code_b) in parsed.items():
            if event_ticker in used or frozenset({code_a, code_b}) != pair:
                continue
            delta = abs((game_date - et_date).days)
            if delta <= 1 and (best is None or delta < best[0]):
                best = (delta, event_ticker)
        if best is None:
            unmatched_odds.append(event)
            continue

        event_ticker = best[1]
        used.add(event_ticker)
        names = {home_code: event.home_team, away_code: event.away_team}
        resolved: list[dict] = []
        unresolved: list[dict] = []
        for market in by_event[event_ticker]:
            code = _market_yes_team_code(market, (home_code, away_code))
            if code:
                resolved.append({**market, "_yes_team": names[code]})
            else:
                unresolved.append(market)
        games.append(MatchedGame(event, event_ticker, event_ticker.split("-")[0], resolved, unresolved))

    unmatched_kalshi += [e for e in parsed if e not in used]
    return games, unmatched_odds, unmatched_kalshi


# =============================================================================
# Account state, paper ledger, risk manager
# =============================================================================
@dataclass
class AccountState:
    equity: Decimal  # cash + position value (+ resting-order reserve) - the Kelly bankroll
    cash: Decimal  # available to trade
    exposure: Decimal  # cost of open positions + resting orders
    tickers: set[str]
    events: set[str]
    source: str  # "kalshi" or "paper"

    def has_exposure(self, ticker: str, event_ticker: str) -> bool:
        return ticker in self.tickers or event_ticker in self.events

    def register(self, plan: "OrderPlan") -> None:
        self.cash -= plan.total_cost
        self.exposure += plan.total_cost
        self.tickers.add(plan.ticker)
        self.events.add(plan.event_ticker)


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
        events={event_from_ticker(t) for t in tickers if t},
        source="kalshi",
    )


class PaperLedger:
    """Remembers simulated orders so duplicate/exposure rules work across DRY_RUN scans."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.PAPER_LEDGER_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                log.warning("Paper ledger unreadable; starting empty.")

    def prune(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=config.PAPER_LEDGER_TTL_HOURS)
        self.entries = [e for e in self.entries if datetime.fromisoformat(e["ts"]) >= cutoff]

    def record(self, plan: "OrderPlan", now: datetime) -> None:
        self.entries.append({
            "ts": now.isoformat(), "ticker": plan.ticker, "event_ticker": plan.event_ticker,
            "side": plan.side, "price": str(plan.price), "contracts": plan.contracts,
            "route": plan.route, "stake": str(plan.total_cost),
        })
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")

    def account(self, now: datetime) -> AccountState:
        self.prune(now)
        exposure = sum((Decimal(e["stake"]) for e in self.entries), me.ZERO)
        return AccountState(
            equity=config.PAPER_BANKROLL,
            cash=config.PAPER_BANKROLL - exposure,
            exposure=exposure,
            tickers={e["ticker"] for e in self.entries},
            events={e["event_ticker"] for e in self.entries},
            source="paper",
        )


class RiskManager:
    """
    Non-negotiable circuit breakers, re-checked immediately before any order is sent:
      * daily drawdown stop (sticky for the rest of the day once tripped)
      * duplicate position / resting-order prevention (per ticker and per game)
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

    def check_halt(self, account: AccountState) -> bool:
        """Update the day's drawdown from current equity; True means trading is halted."""
        state = self._load()
        key = f"{account.source}:{config.KALSHI_ENV}"
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
        if account.has_exposure(plan.ticker, plan.event_ticker):
            return Decision.SKIP_DUPLICATE, "open position or resting order already exists for this game/ticker"
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
    fair_side: Decimal
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def post_only(self) -> bool:
        return self.route == "maker"

    @property
    def fill_or_kill(self) -> bool:
        return self.route == "taker"


@dataclass
class Evaluation:
    decision: Decision
    reason: str
    rec: dict[str, Any]
    plan: OrderPlan | None = None


class ExecutionEngine:
    def __init__(
        self,
        kalshi: KalshiClient,
        journal: TradeJournal,
        risk: RiskManager,
        ledger: PaperLedger,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.kalshi = kalshi
        self.journal = journal
        self.risk = risk
        self.ledger = ledger
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.counts: Counter[str] = Counter()
        self.buys: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ logging helpers
    def _log(self, decision: Decision, reason: str, rec: dict[str, Any]) -> None:
        self.counts[decision.value] += 1
        self.journal.log(decision, reason, rec)

    def log_unmatched_odds(self, event: OddsEvent, snapshot: OddsSnapshot) -> None:
        now = self._now()
        self._log(Decision.SKIP_NO_MATCH, "no Kalshi NBA game market found for this Odds API game", {
            "event_id": event.event_id,
            "team_yes": f"{event.away_team} @ {event.home_team}",
            "commence_time": event.commence_time,
            "minutes_to_tipoff": (event.commence_time - now).total_seconds() / 60,
            "odds_age_seconds": snapshot.age_seconds(now),
        })

    def log_unmatched_kalshi(self, event_ticker: str) -> None:
        self._log(Decision.SKIP_NO_MATCH, "Kalshi event has no matching Odds API game", {
            "kalshi_event_ticker": event_ticker,
        })

    # ------------------------------------------------------------------ per-game pipeline
    def process_game(self, game: MatchedGame, snapshot: OddsSnapshot, account: AccountState) -> None:
        now = self._now()
        event = game.odds_event
        minutes_to_tipoff = Decimal(str((event.commence_time - now).total_seconds())) / 60
        base: dict[str, Any] = {
            "event_id": event.event_id,
            "kalshi_event_ticker": game.kalshi_event_ticker,
            "commence_time": event.commence_time,
            "minutes_to_tipoff": round(minutes_to_tipoff, 1),
            "odds_age_seconds": round(Decimal(str(snapshot.age_seconds(now))), 1),
            "bankroll": account.equity,
            "cash": account.cash,
        }

        def log_all(decision: Decision, reason: str, extra: dict | None = None) -> None:
            for market in game.markets:
                self._log(decision, reason, {**base, **(extra or {}), "kalshi_ticker": market["ticker"],
                                             "team_yes": market["_yes_team"]})

        for market in game.unresolved:
            self._log(Decision.SKIP_NO_MATCH, "could not tell which team this market's YES side refers to",
                      {**base, "kalshi_ticker": market.get("ticker", "")})
        if not game.markets:
            return

        # --- global gates (no API calls) -------------------------------------------------
        if self.risk.check_halt(account):
            log_all(Decision.HALT_DRAWDOWN, self.risk.last_status.get("reason", "trading halted"))
            return
        if config.SKIP_LIVE_GAMES and minutes_to_tipoff < config.MIN_MINUTES_BEFORE_TIPOFF:
            log_all(Decision.SKIP_LIVE, f"game starts in {minutes_to_tipoff:.1f} min (or already started)")
            return
        if snapshot.age_seconds(now) > config.MAX_ODDS_AGE_SECONDS:
            log_all(Decision.SKIP_STALE, f"odds snapshot is {snapshot.age_seconds(now):.0f}s old "
                                         f"(> {config.MAX_ODDS_AGE_SECONDS}s)")
            return

        fresh, stale = event.fresh_quotes(now, config.MAX_BOOK_UPDATE_AGE_SECONDS)
        if not fresh:
            if stale:
                oldest = max((now - q.last_update).total_seconds() for q in stale)
                log_all(Decision.SKIP_STALE, f"all {len(stale)} books older than "
                                             f"{config.MAX_BOOK_UPDATE_AGE_SECONDS}s (oldest {oldest:.0f}s)")
            else:
                log_all(Decision.SKIP_NO_ODDS, "no bookmaker quotes for this game")
            return

        consensus = me.build_consensus(
            [(q.book_key, q.prices) for q in fresh], (event.home_team, event.away_team))
        if consensus is None:
            log_all(Decision.SKIP_NO_ODDS, "no usable two-way moneyline quotes")
            return
        base.update({
            "books_used": consensus.books_used,
            "sharp_books": consensus.sharp_books_used,
            "sharp_weight_share": round(consensus.sharp_weight_share, 4),
            "consensus_dispersion": round(consensus.dispersion, 5),
            "stale_books_dropped": len(stale),
        })
        if len(consensus.books_used) < config.MIN_BOOKS_FOR_CONSENSUS:
            log_all(Decision.SKIP_CONSENSUS, f"only {len(consensus.books_used)} fresh book(s); "
                                              f"need {config.MIN_BOOKS_FOR_CONSENSUS}")
            return
        if config.REQUIRE_SHARP_BOOK and not consensus.sharp_books_used:
            log_all(Decision.SKIP_CONSENSUS, "no fresh sharp book in the consensus")
            return

        if any(account.has_exposure(m["ticker"], game.kalshi_event_ticker) for m in game.markets):
            log_all(Decision.SKIP_DUPLICATE, "open position or resting order already exists for this game",
                    {"fair_prob_yes_team": None})
            return

        # --- per-market evaluation (order book API calls) --------------------------------
        fee_schedule = self.kalshi.get_fee_schedule(game.series_ticker)
        evaluations: list[Evaluation] = []
        for market in game.markets:
            fair_yes = consensus.probs[market["_yes_team"]]
            evaluations.append(self._evaluate_market(market, game, fair_yes, fee_schedule, account, base))

        buyable = [e for e in evaluations if e.plan is not None]
        winner = max(buyable, key=lambda e: (e.plan.net_edge, e.plan.total_cost)) if buyable else None
        for evaluation in evaluations:
            if evaluation is winner:
                continue
            if evaluation.plan is not None:  # +EV, but another market in the same game is better
                self._log(Decision.SKIP_DUPLICATE,
                          "same-game candidate with a higher net edge was selected", evaluation.rec)
            else:
                self._log(evaluation.decision, evaluation.reason, evaluation.rec)
        if winner is not None:
            self._execute(winner, account)

    # ------------------------------------------------------------------ market evaluation
    def _evaluate_market(
        self, market: dict, game: MatchedGame, fair_yes: Decimal, fee_schedule: me.FeeSchedule,
        account: AccountState, base: dict[str, Any],
    ) -> Evaluation:
        ticker = market["ticker"]
        rec: dict[str, Any] = {**base, "kalshi_ticker": ticker, "team_yes": market["_yes_team"],
                               "fair_prob_yes_team": round(fair_yes, 5)}

        def done(decision: Decision, reason: str, plan: OrderPlan | None = None) -> Evaluation:
            return Evaluation(decision, reason, rec, plan)

        try:
            book = self.kalshi.get_orderbook(ticker)
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
        if spread > config.MAX_SPREAD:
            return done(Decision.SKIP_SPREAD, f"spread {spread:.2f} > max {config.MAX_SPREAD:.2f}")

        # --- price both sides: resting maker price (never crossing) and taker price at the ask
        sides: list[dict[str, Any]] = []
        for side in ("yes", "no"):
            fair_side = fair_yes if side == "yes" else me.ONE - fair_yes
            bid, ask = book.best_bid(side), book.best_ask(side)
            maker_price = min(bid + config.MAKER_PRICE_IMPROVEMENT, ask - config.TICK)
            maker_price = min(max(maker_price, config.TICK), me.ONE - config.TICK)
            maker_edge = me.net_edge(
                fair_side, maker_price, me.fee_per_contract_continuous(maker_price, "maker", fee_schedule))
            taker_edge = me.net_edge(
                fair_side, ask, me.fee_per_contract_continuous(ask, "taker", fee_schedule))
            sides.append({"side": side, "fair": fair_side, "bid": bid, "ask": ask,
                          "maker_price": maker_price, "maker_edge": maker_edge, "taker_edge": taker_edge})
        rec["details"] = {
            "edge_by_side": {s["side"]: {"maker_price": s["maker_price"], "maker_edge": round(s["maker_edge"], 5),
                                         "taker_edge": round(s["taker_edge"], 5)} for s in sides},
            "fee_type": fee_schedule.fee_type, "fee_multiplier": fee_schedule.multiplier,
        }

        best = max(sides, key=lambda s: s["maker_edge"])
        side, fair_side, ask = best["side"], best["fair"], best["ask"]
        rec.update(side=side, fair_prob_side=round(fair_side, 5), order_price=best["maker_price"],
                   net_edge=round(best["maker_edge"], 5))
        if best["maker_edge"] < config.MIN_NET_EDGE:
            return done(Decision.PASS, f"best net edge {best['maker_edge']:.4f} < min {config.MIN_NET_EDGE:.4f}")

        # --- liquidity: complementary depth at the best ask
        depth = book.ask_depth(side)
        rec["ask_depth"] = depth
        if depth < config.MIN_ASK_DEPTH_CONTRACTS:
            return done(Decision.SKIP_LIQUIDITY,
                        f"only {depth} contracts at the best ask (need >= {config.MIN_ASK_DEPTH_CONTRACTS})")

        # --- routes in priority order: maker-first, taker only for a large, fee-adjusted edge
        routes: list[tuple[str, Decimal]] = []
        if config.ALLOW_TAKER_ORDERS and best["taker_edge"] >= max(config.TAKER_OVERRIDE_MIN_EDGE, config.MIN_NET_EDGE):
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
            if exact_edge < config.MIN_NET_EDGE:
                last_decision = Decision.PASS
                last_reason = (f"{route}: net edge {exact_edge:.4f} at {sizing.contracts} contracts "
                               f"(after exact fee rounding) < min {config.MIN_NET_EDGE:.4f}")
                continue
            if route == "taker" and depth < sizing.contracts * config.DEPTH_COVERAGE_MULTIPLE:
                last_decision = Decision.SKIP_LIQUIDITY
                last_reason = f"taker: {depth} contracts at the ask < {sizing.contracts} needed"
                continue
            plan = OrderPlan(
                ticker=ticker, event_ticker=game.kalshi_event_ticker, side=side, price=price,
                contracts=sizing.contracts, route=route, fee_total=sizing.fee,
                total_cost=sizing.total_cost, net_edge=exact_edge, fair_side=fair_side)
            rec["client_order_id"] = plan.client_order_id
            return done(Decision.BUY, f"{route} order, net edge {exact_edge:.4f}", plan)
        return done(last_decision, last_reason)

    # ------------------------------------------------------------------ order placement
    def _execute(self, evaluation: Evaluation, account: AccountState) -> None:
        plan, rec = evaluation.plan, evaluation.rec
        assert plan is not None
        decision, why = self.risk.pre_trade_check(plan, account)
        if decision is not None:
            self._log(decision, why, rec)
            return

        if config.DRY_RUN:
            self.ledger.record(plan, self._now())
            rec["api_status"] = "simulated"
            rec["api_response"] = {"mode": "DRY_RUN", "simulated": True, "note": "no order sent to Kalshi"}
        elif config.live_trading_armed():
            try:
                response = self.kalshi.place_order(
                    ticker=plan.ticker, side=plan.side, count=plan.contracts, price=plan.price,
                    client_order_id=plan.client_order_id, post_only=plan.post_only,
                    fill_or_kill=plan.fill_or_kill)
            except Exception as exc:  # noqa: BLE001
                rec["api_status"] = "order_error"
                rec["api_response"] = str(exc)
                self._log(Decision.ERROR, f"order placement failed: {exc}", rec)
                return
            rec["api_status"] = "accepted"
            rec["api_response"] = response
            order = response.get("order") if isinstance(response, dict) else None
            if isinstance(order, dict):
                rec["order_id"] = order.get("order_id")
        else:
            self._log(Decision.SKIP_RISK,
                      f"DRY_RUN is False but {config.LIVE_ARM_ENV_VAR} is not set to the arming phrase", rec)
            return

        account.register(plan)
        self.buys.append({"ticker": plan.ticker, "side": plan.side, "contracts": plan.contracts,
                          "price": plan.price, "route": plan.route, "net_edge": plan.net_edge,
                          "stake": plan.total_cost})
        self._log(Decision.BUY, evaluation.reason, rec)

    # ------------------------------------------------------------------ scan driver
    def process_all(self, games: list[MatchedGame], snapshot: OddsSnapshot, account: AccountState) -> None:
        for game in games:
            try:
                self.process_game(game, snapshot, account)
            except Exception as exc:  # noqa: BLE001 - one bad game must not kill the scan
                log.exception("Unhandled error processing %s", game.kalshi_event_ticker)
                self._log(Decision.ERROR, f"unhandled error: {exc}",
                          {"kalshi_event_ticker": game.kalshi_event_ticker, "event_id": game.odds_event.event_id})
