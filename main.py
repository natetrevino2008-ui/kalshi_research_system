"""
main.py - TrueLine entry point: ingestion -> matching -> pricing -> gates -> sizing -> picks -> (paper|real) execution.

Usage
    python main.py --mode demo               # offline demo with synthetic games; touches no network
    python main.py                           # --mode paper: real prices, simulated money, auto paper-trades picks
    python main.py --markets moneyline,spread,total,prop      # choose market types (props cost extra Odds API credits)
    python main.py --mode paper --settle     # settle open paper trades against Kalshi, then exit
    python main.py --mode live               # REAL orders, interactive: asks you to type YES for every single order
    python main.py --loop --interval 600 -v  # rescan every N seconds (paper only; mind your Odds API credits)

Real money (--mode live) needs ALL of: KALSHI_API_KEY + KALSHI_PRIVATE_KEY, KALSHI_ARM_LIVE=YES_I_WANT_REAL_ORDERS,
and a typed confirmation for each order. KALSHI_ENV=demo (the default) uses Kalshi's play-money environment;
set KALSHI_ENV=prod only when you mean it.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Iterable

import config
import math_engine as me
from execution_engine import (
    AccountState, Decision, ExecutionEngine, Pick, RiskManager, SettleReport, TradeJournal, TradeLedger,
    build_live_account, execute_paper, match_games, place_live_order, prop_event_filter, settle_trades,
)
from kalshi_client import KalshiClient, KalshiScan, OrderBook, parse_market
from odds_client import BookQuote, OddsAPIClient, OddsEvent, OddsSnapshot

log = logging.getLogger("main")


# =============================================================================
# One scan
# =============================================================================
@dataclass
class ScanResult:
    snapshot: OddsSnapshot
    engine: ExecutionEngine
    picks: list[Pick]
    executed: list[tuple[Pick, bool, str]] = field(default_factory=list)
    kalshi_errors: list[str] = field(default_factory=list)
    kalshi_markets: int = 0
    unresolved: int = 0
    games: int = 0
    unmatched_odds: int = 0
    unmatched_kalshi: int = 0
    run_id: str = ""
    account: AccountState | None = None  # account as it was when the scan started


def run_scan(
    kalshi,
    fetch_snapshot: Callable[..., OddsSnapshot],
    *,
    journal: TradeJournal,
    risk: RiskManager,
    ledger: TradeLedger,
    mode: str = "paper",
    categories: Iterable[str] | None = None,
    auto_paper: bool = False,
    say: Callable[[str], None] = print,
) -> ScanResult:
    """
    Scan, price and PROPOSE picks. Paper/demo picks are recorded only if auto_paper is True; real-money
    picks are never sent from here - the caller must confirm each one (see place_live_order).
    """
    categories = tuple(categories) if categories is not None else config.enabled_categories()
    labels = ", ".join(config.CATEGORIES[c].label for c in categories)

    # 1. Ingest: Kalshi markets for every enabled category (public, free), normalised to exact lines
    kscan: KalshiScan = kalshi.get_specs(categories)
    say(f"Kalshi: {len(kscan.specs)} markets parsed ({len(kscan.unresolved)} unparsed) for {labels}")
    for error in kscan.errors:
        say(f"  Kalshi fetch problem: {error}")

    # 2. Ingest: Odds API. ONE bulk call for h2h/spreads/totals; per-event calls ONLY for props, and only
    #    for games Kalshi lists prop markets for.
    snapshot = fetch_snapshot(categories, prop_event_filter(kscan.specs))
    say(f"Odds API: {len(snapshot.events)} games | credits used this scan: {snapshot.scan_cost} "
        f"| remaining: {snapshot.credits_remaining}")
    for warning in snapshot.warnings:
        say(f"  Odds API note: {warning}")

    # 3. Account state: ledger-derived paper account, or the real Kalshi balance/positions/orders
    now = datetime.now(timezone.utc)
    account: AccountState = build_live_account(kalshi) if mode == "live" else ledger.account(now, mode)
    say(f"Account ({account.source}): equity ${account.equity:.2f} | cash ${account.cash:.2f} "
        f"| exposure ${account.exposure:.2f}")

    start_account = replace(account, tickers=set(account.tickers), groups=set(account.groups))

    # 4. Match games, then run the decision pipeline
    games, unmatched_odds, unmatched_kalshi = match_games(snapshot.events, kscan.specs)
    engine = ExecutionEngine(kalshi, journal, risk)
    for error in kscan.errors:
        engine._log(Decision.ERROR, f"Kalshi fetch failed: {error}", {})
    engine.log_unresolved(kscan.unresolved)
    for event in unmatched_odds:
        engine.log_unmatched_odds(event, snapshot)
    for game_label in unmatched_kalshi:
        engine.log_unmatched_kalshi(game_label)
    engine.process_all(games, snapshot, account)

    result = ScanResult(
        snapshot=snapshot, engine=engine, picks=list(engine.picks), kalshi_errors=list(kscan.errors),
        kalshi_markets=len(kscan.specs), unresolved=len(kscan.unresolved), games=len(games),
        unmatched_odds=len(unmatched_odds), unmatched_kalshi=len(unmatched_kalshi), run_id=journal.run_id, account=start_account,
    )

    # 5. Paper/demo auto-trading (never for real money)
    if auto_paper and mode in ("paper", "demo"):
        for pick in engine.picks:
            ok, why = execute_paper(pick, ledger=ledger, journal=journal, risk=risk, mode=mode)
            result.executed.append((pick, ok, why))

    # 6. Summary
    say("\n---- Scan summary ----")
    say(f"Matched games: {len(games)} | unmatched Odds API games: {len(unmatched_odds)} "
        f"| unmatched Kalshi games: {len(unmatched_kalshi)} | order-book calls: {engine.book_calls}")
    for decision, count in sorted(engine.counts.items()):
        say(f"  {decision:<16} {count}")
    if engine.picks:
        say("\nPICKS:")
        for pick in engine.picks:
            plan = pick.plan
            say(f"  [{plan.market_type}] {plan.label}: BUY {plan.contracts} {plan.side.upper()} {plan.ticker} @ "
                f"${plan.price:.2f} [{plan.route}] edge {plan.net_edge:.2%} | max loss ${plan.total_cost:.2f}")
    status = risk.last_status
    if status:
        say(f"\nRisk: day-start equity ${status['start_equity']:.2f} | drawdown {status['drawdown']:.2%} "
            f"(halt at {config.MAX_DAILY_DRAWDOWN:.0%}) | halted: {status['halted']}")
    say(f"Journal: {journal.csv_path}")
    return result


# =============================================================================
# Offline mock (synthetic Odds API + Kalshi data, exercises every decision path and every market type)
# =============================================================================
def _american_from_prob(prob: Decimal) -> Decimal:
    if prob >= Decimal("0.5"):
        return -((prob / (1 - prob)) * 100).to_integral_value()
    return (((1 - prob) / prob) * 100).to_integral_value()


_MOCK_BOOKS = (("pinnacle", "0"), ("lowvig", "0.004"), ("draftkings", "-0.008"), ("fanduel", "-0.005"), ("betmgm", "0.006"))


def _mock_quote(book: str, shift: str, market: str, pairs, updated: datetime) -> BookQuote:
    """pairs: [(yes_key, no_key, p_yes)] -> a BookQuote with ~4.5% two-way overround on every pair."""
    vig = Decimal("1.0225")
    selections = {}
    for yes_key, no_key, p_yes in pairs:
        p = min(max(Decimal(p_yes) + Decimal(shift), Decimal("0.02")), Decimal("0.98"))
        selections[yes_key] = _american_from_prob(p * vig)
        selections[no_key] = _american_from_prob((1 - p) * vig)
    return BookQuote(book_key=book, book_title=book, last_update=updated, market=market, selections=selections)


class MockKalshi:
    """Duck-typed stand-in for KalshiClient used by the demo mode. Never sends orders."""

    has_credentials = False

    def __init__(self) -> None:
        self.raw_markets: dict[str, list[dict]] = defaultdict(list)
        self.books: dict[str, OrderBook] = {}

    def _add(self, series: str, ticker: str, event_ticker: str, fields: dict, yes_levels, no_levels) -> None:
        raw = {"ticker": ticker, "event_ticker": event_ticker, "status": "open", **fields,
               "yes_bid_dollars": f"{yes_levels[-1][0]:.4f}", "no_bid_dollars": f"{no_levels[-1][0]:.4f}"}
        self.raw_markets[series].append(raw)
        self.books[ticker] = OrderBook(ticker, tuple(yes_levels), tuple(no_levels))

    def add_moneyline(self, event_ticker: str, away: str, home: str, bid: str, ask: str, ask_depth: int, bid_depth: int = 40):
        yes_bid, yes_ask = Decimal(bid), Decimal(ask)
        home_yes = ((yes_bid, Decimal(bid_depth)),)
        home_no = ((1 - yes_ask, Decimal(ask_depth)),)
        for code, yes_levels, no_levels in ((home, home_yes, home_no), (away, home_no, home_yes)):
            self._add("KXNBAGAME", f"{event_ticker}-{code}", event_ticker,
                      {"yes_sub_title": config.NBA_TEAMS[code][1], "title": f"{config.NBA_TEAMS[code][1]} winner?"},
                      yes_levels, no_levels)

    def add_quote(self, series: str, event_ticker: str, suffix: str, fields: dict, bid: str, ask: str,
                  ask_depth: int = 40, bid_depth: int = 40) -> None:
        self._add(series, f"{event_ticker}-{suffix}", event_ticker, fields,
                  ((Decimal(bid), Decimal(bid_depth)),), ((1 - Decimal(ask), Decimal(ask_depth)),))

    def get_markets(self, series_ticker: str, status: str = "open") -> list[dict]:
        return list(self.raw_markets.get(series_ticker, []))

    def get_specs(self, categories: Iterable[str]) -> KalshiScan:
        scan = KalshiScan()
        for category in categories:
            for series in config.kalshi_series_for(category):
                raws = self.get_markets(series)
                scan.raw_counts[series] = len(raws)
                for raw in raws:
                    spec, why = parse_market(raw)
                    if spec is not None:
                        scan.specs.append(spec)
                    else:
                        scan.unresolved.append((raw, why))
        return scan

    def get_orderbook(self, ticker: str) -> OrderBook:
        return self.books[ticker]

    def get_fee_schedule(self, series_ticker: str) -> me.FeeSchedule:
        return me.FeeSchedule("quadratic", Decimal("1"))

    def place_order(self, **_kwargs):  # pragma: no cover - guard
        raise RuntimeError("MockKalshi must never place orders.")


def build_mock_world() -> tuple[MockKalshi, Callable[..., OddsSnapshot]]:
    now = datetime.now(timezone.utc)
    kalshi = MockKalshi()
    events: list[OddsEvent] = []
    fresh = now - timedelta(seconds=15)
    stale = now - timedelta(minutes=10)
    K = me.SelectionKey

    # (away, home, p_home, commence_offset_min, book_update, kalshi moneyline (bid, ask, ask_depth) or None, scenario)
    scenarios = [
        ("LAL", "HOU", "0.62", 240, fresh, ("0.53", "0.55", 40), "edge -> resting maker BUY"),
        ("BOS", "NYK", "0.70", 240, fresh, ("0.58", "0.60", 30), "big edge -> taker BUY"),
        ("MIA", "DEN", "0.55", 240, fresh, ("0.54", "0.56", 40), "fairly priced -> PASS"),
        ("CHI", "MIL", "0.60", 240, fresh, ("0.45", "0.53", 40), "wide spread -> SKIP_SPREAD"),
        ("DAL", "PHX", "0.62", 240, fresh, ("0.53", "0.55", 2), "thin ask depth -> SKIP_LIQUIDITY"),
        ("ATL", "BKN", "0.62", 240, stale, ("0.53", "0.55", 40), "stale odds -> SKIP_STALE"),
        ("TOR", "UTA", "0.58", 240, fresh, None, "no Kalshi market -> SKIP_NO_MATCH"),
        ("GSW", "SAC", "0.62", -30, fresh, ("0.53", "0.55", 40), "already started -> SKIP_LIVE"),
        ("MEM", "LAL", "0.66", 300, fresh, None, "spread only: 'Los Angeles L' alias"),
    ]
    for index, (away, home, p_home, offset_min, updated, ml_quote, _label) in enumerate(scenarios):
        commence = now + timedelta(minutes=offset_min)
        home_name, away_name = " ".join(config.NBA_TEAMS[home]), " ".join(config.NBA_TEAMS[away])
        date_code = commence.astimezone(config.GAME_TZ).strftime("%y%b%d").upper()
        code = f"{date_code}{away}{home}"
        p = Decimal(p_home)

        # book quotes: moneyline for every game
        by_market: dict[str, list] = {"h2h": [(K("h2h", home, "win", None), K("h2h", away, "win", None), p)]}
        players: dict[str, dict[str, str]] = {}

        if ml_quote:
            kalshi.add_moneyline(f"KXNBAGAME-{code}", away, home, *ml_quote)

        if (away, home) == ("LAL", "HOU"):
            # Spread: HOU -3.5 is the book line. Kalshi "Houston wins by over 3.5" matches it; 9.5 has no book line.
            by_market["spreads"] = [(K("spreads", "HOU", "cover", Decimal("-3.5")), K("spreads", "LAL", "cover", Decimal("3.5")), Decimal("0.52"))]
            ev = f"KXNBASPREAD-{code}"
            kalshi.add_quote("KXNBASPREAD", ev, "HOU4", {"title": "Houston wins by over 3.5 Points?", "yes_sub_title": "Houston wins by over 3.5 Points",
                                                         "floor_strike": 3.5, "strike_type": "greater"}, "0.43", "0.45")
            kalshi.add_quote("KXNBASPREAD", ev, "HOU10", {"title": "Houston wins by over 9.5 Points?", "yes_sub_title": "Houston wins by over 9.5 Points",
                                                          "floor_strike": 9.5, "strike_type": "greater"}, "0.20", "0.22")
            # Total: book line 226.5. Kalshi 226.5 matches; 232.5 has no book line.
            by_market["totals"] = [(K("totals", "game", "over", Decimal("226.5")), K("totals", "game", "under", Decimal("226.5")), Decimal("0.50"))]
            ev = f"KXNBATOTAL-{code}"
            kalshi.add_quote("KXNBATOTAL", ev, "227", {"title": "Total points over 226.5?", "yes_sub_title": "Over 226.5 points scored",
                                                       "floor_strike": 226.5, "strike_type": "greater"}, "0.44", "0.46")
            kalshi.add_quote("KXNBATOTAL", ev, "233", {"title": "Total points over 232.5?", "yes_sub_title": "Over 232.5 points scored",
                                                       "floor_strike": 232.5, "strike_type": "greater"}, "0.20", "0.22")
            # Props: book says "Alperen Şengün" (accents), Kalshi says "Alperen Sengun" - must still match.
            by_market["player_points"] = [
                (K("player_points", me.normalize_player_name("Alperen Şengün"), "over", Decimal("19.5")),
                 K("player_points", me.normalize_player_name("Alperen Şengün"), "under", Decimal("19.5")), Decimal("0.56")),
                (K("player_points", me.normalize_player_name("Fred VanVleet"), "over", Decimal("14.5")),
                 K("player_points", me.normalize_player_name("Fred VanVleet"), "under", Decimal("14.5")), Decimal("0.50")),
            ]
            players["player_points"] = {me.normalize_player_name("Alperen Şengün"): "Alperen Şengün",
                                        me.normalize_player_name("Fred VanVleet"): "Fred VanVleet"}
            ev = f"KXNBAPTS-{code}"
            kalshi.add_quote("KXNBAPTS", ev, "HOUASENGUN28-20", {"title": "Alperen Sengun: 20+ points", "yes_sub_title": "Alperen Sengun: 20+",
                                                                  "floor_strike": 19.5, "strike_type": "greater"}, "0.47", "0.49")
            kalshi.add_quote("KXNBAPTS", ev, "HOUFVANVLEET5-15", {"title": "Fred VanVleet: 15+ points", "yes_sub_title": "Fred VanVleet: 15+",
                                                                   "floor_strike": 14.5, "strike_type": "greater"}, "0.49", "0.51")
            kalshi.add_quote("KXNBAPTS", ev, "HOUASENGUN28-30", {"title": "Alperen Sengun: 30+ points", "yes_sub_title": "Alperen Sengun: 30+",
                                                                  "floor_strike": 29.5, "strike_type": "greater"}, "0.06", "0.08")

        if (away, home) == ("MEM", "LAL"):
            # Kalshi shortens the Lakers to "Los Angeles L"; LAL -5.5 is the book line
            by_market["spreads"] = [(K("spreads", "LAL", "cover", Decimal("-5.5")), K("spreads", "MEM", "cover", Decimal("5.5")), Decimal("0.50"))]
            kalshi.add_quote("KXNBASPREAD", f"KXNBASPREAD-{code}", "LAL6", {"title": "Los Angeles L wins by over 5.5 Points?",
                             "yes_sub_title": "Los Angeles L wins by over 5.5 Points", "floor_strike": 5.5, "strike_type": "greater"}, "0.50", "0.52")

        quotes = []
        for book, shift in _MOCK_BOOKS:
            for market, pairs in by_market.items():
                quotes.append(_mock_quote(book, shift, market, pairs, updated))
        events.append(OddsEvent(f"mock-{index}", home_name, away_name, commence, quotes, players))

    def fetch(categories: Iterable[str] | None = None, event_filter=None) -> OddsSnapshot:
        cats = tuple(categories) if categories is not None else config.enabled_categories()
        allowed = set(config.bulk_odds_markets(cats)) | set(config.prop_odds_markets(cats))
        shown = [OddsEvent(e.event_id, e.home_team, e.away_team, e.commence_time,
                           [q for q in e.quotes if q.market in allowed], dict(e.players)) for e in events]
        prop_games = sum(1 for e in shown if any(q.market.startswith("player_") for q in e.quotes))
        cost = len(config.bulk_odds_markets(cats)) + len(config.prop_odds_markets(cats)) * prop_games
        return OddsSnapshot(fetched_at=datetime.now(timezone.utc), events=shown, credits_remaining=480 - cost,
                            credits_used=20 + cost, last_call_cost=cost, scan_cost=cost, api_calls=1 + prop_games)

    return kalshi, fetch


# =============================================================================
# World factory (used by the CLI and the Streamlit app)
# =============================================================================
def make_kalshi(mode: str) -> KalshiClient:
    """Paper reads PRODUCTION market data (public, read-only); live uses KALSHI_ENV (where real orders go)."""
    if mode == "paper":
        return KalshiClient(base_url=config.PAPER_DATA_BASE_URL)
    return KalshiClient()


def make_world(mode: str):
    """Returns (kalshi client, fetch_snapshot(categories, event_filter))."""
    if mode == "demo":
        return build_mock_world()
    return make_kalshi(mode), OddsAPIClient().fetch_nba_odds


def settle_scope(ledger: TradeLedger, modes: Iterable[str]) -> SettleReport:
    """Settle open trades per mode, each against the right exchange (paper -> data env, live -> KALSHI_ENV)."""
    total = SettleReport()
    for mode in modes:
        report = settle_trades(ledger, None if mode == "demo" else make_kalshi(mode), modes=(mode,),
                               simulate=(mode == "demo"))
        total.checked += report.checked
        total.settled += report.settled
        total.updated += report.updated
        total.canceled += report.canceled
        total.errors.extend(report.errors)
    return total


# =============================================================================
# CLI
# =============================================================================
def print_banner(mode: str, categories: tuple[str, ...]) -> None:
    print("=" * 70)
    print(f" {config.APP_NAME} | mode: {config.MODE_LABELS[mode]}")
    if mode != "demo":
        if mode == "live":
            env_note = "PRODUCTION - REAL MONEY" if config.is_prod() else "demo environment (play money)"
            print(f" Kalshi env: {config.KALSHI_ENV} ({config.KALSHI_BASE_URL}) -> {env_note}")
        else:
            print(f" Kalshi market data: {config.PAPER_DATA_ENV} ({config.PAPER_DATA_BASE_URL}), read-only")
    print(f" markets: {', '.join(config.CATEGORIES[c].label for c in categories)} "
          f"| est. Odds API credits/scan (10 games): {config.estimate_credits(categories)}")
    print(f" devig: {config.DEVIG_METHOD} | min edge {config.MIN_NET_EDGE:.1%} | Kelly x{config.KELLY_FRACTION}")
    print(f" caps: {config.MAX_BET_FRACTION:.0%} of equity / ${config.MAX_BET_DOLLARS} per bet | daily stop -{config.MAX_DAILY_DRAWDOWN:.0%}")
    print("=" * 70)
    if config.TZ_FALLBACK_USED:
        print("WARNING: time-zone database not found (pip install tzdata); using fixed UTC offsets.")


def _confirm_live(result: ScanResult, kalshi, journal, risk, ledger) -> None:
    """Interactive real-money flow: every order needs its own typed YES."""
    if not result.picks:
        print("\nNo qualifying picks.")
        return
    for pick in result.picks:
        plan = pick.plan
        print(f"\n[{plan.market_type}] {plan.label}\n  BUY {plan.contracts} {plan.side.upper()} {plan.ticker} @ ${plan.price:.2f} "
              f"[{plan.route}] edge {plan.net_edge:.2%} | max loss ${plan.total_cost:.2f}")
        if input("  Type YES to place this REAL order (anything else skips): ").strip() != "YES":
            print("  skipped.")
            continue
        outcome = place_live_order(pick, kalshi=kalshi, journal=journal, risk=risk, ledger=ledger, confirmed=True)
        print(f"  {'OK' if outcome.ok else 'NOT PLACED'}: {outcome.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrueLine - Kalshi NBA +EV scanner / execution engine")
    parser.add_argument("--mode", choices=config.MODES, default="paper", help="demo | paper (default) | live")
    parser.add_argument("--mock", action="store_true", help="alias for --mode demo")
    parser.add_argument("--markets", default=None,
                        help="comma list of moneyline,spread,total,prop (default: config.ENABLED_CATEGORIES)")
    parser.add_argument("--no-auto", action="store_true", help="paper/demo: record picks only, do not auto paper-trade them")
    parser.add_argument("--settle", action="store_true", help="settle open trades against Kalshi and exit")
    parser.add_argument("--loop", action="store_true", help="rescan continuously (not allowed in live mode)")
    parser.add_argument("--interval", type=int, default=config.POLL_INTERVAL_SECONDS, help="seconds between scans")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    mode = "demo" if args.mock else args.mode

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.markets:
        wanted = {m.strip() for m in args.markets.split(",") if m.strip()}
        unknown = wanted - set(config.CATEGORIES)
        if unknown:
            print(f"ERROR: unknown market type(s): {', '.join(sorted(unknown))}. Choose from {', '.join(config.CATEGORIES)}.")
            return 2
        config.ENABLED_CATEGORIES.update({key: key in wanted for key in config.CATEGORIES})
    categories = config.enabled_categories()
    print_banner(mode, categories)

    paths = config.mode_paths(mode)
    if mode == "demo":
        if not args.settle:  # every demo scan starts from a clean sandbox (settling keeps the trades)
            for stale_file in (paths.ledger_csv, paths.risk_state):
                stale_file.unlink(missing_ok=True)
    else:
        problems = config.validate_config(categories, mode)
        if problems:
            print("Configuration problems:\n  - " + "\n  - ".join(problems))
            return 2
    if mode == "live":
        if not config.live_trading_armed():
            print(f"ERROR: live mode needs {config.LIVE_ARM_ENV_VAR}={config.LIVE_ARM_PHRASE} in the environment.")
            return 2
        if args.loop:
            print("ERROR: live mode is interactive (you confirm every order); --loop is not allowed.")
            return 2

    kalshi, fetch_snapshot = make_world(mode)
    if mode == "live" and not kalshi.has_credentials:
        print("ERROR: live mode needs Kalshi credentials (KALSHI_API_KEY + private key).")
        return 2
    if mode == "paper" and not kalshi.has_credentials:
        print("Note: no Kalshi credentials found; using public market data only (fine for paper trading).")

    ledger = TradeLedger(paths.ledger_csv)
    risk = RiskManager(paths.risk_state)
    try:
        if args.settle:
            report = settle_scope(ledger, (mode,))
            print(f"Checked {report.checked} open trade(s): {report.settled} settled, {report.updated} updated, "
                  f"{report.canceled} canceled.")
            for error in report.errors:
                print(f"  ! {error}")
            stats = ledger.stats(mode)
            print(f"Settled {stats['settled']} | wins {stats['wins']} | P&L ${stats['pnl']:.2f} | ROI "
                  f"{'n/a' if stats['roi'] is None else format(stats['roi'], '.1%')}")
            return 0
        while True:
            journal = TradeJournal(paths.journal_csv, paths.journal_jsonl, mode=mode)
            result = run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger, mode=mode,
                              categories=categories, auto_paper=(mode in ("paper", "demo") and not args.no_auto))
            if mode == "live":
                _confirm_live(result, kalshi, journal, risk, ledger)
            if not args.loop:
                break
            remaining = result.snapshot.credits_remaining
            if remaining is not None and remaining < config.MIN_ODDS_CREDITS_REMAINING:
                print(f"Stopping: only {remaining} Odds API credits left (floor {config.MIN_ODDS_CREDITS_REMAINING}).")
                break
            print(f"\nSleeping {args.interval}s (Ctrl+C to stop)...\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:  # noqa: BLE001
        log.exception("Scan failed")
        print(f"ERROR: scan failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
