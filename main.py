"""
main.py - Entry point: ingestion -> matching -> pricing -> gates -> sizing -> (simulated) execution.

Usage
    python main.py                 # one scan, DRY_RUN (default, simulated orders)
    python main.py --mock          # offline demo with synthetic games; touches no network
    python main.py --loop          # rescan every POLL_INTERVAL_SECONDS (mind your Odds API credits)
    python main.py --loop --interval 600 -v

Going live requires BOTH: DRY_RUN = False in config.py AND the environment variable
KALSHI_ARM_LIVE=YES_I_WANT_REAL_ORDERS. Start on KALSHI_ENV=demo.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

import config
import math_engine as me
from execution_engine import (
    AccountState, ExecutionEngine, PaperLedger, RiskManager, TradeJournal,
    build_live_account, match_games,
)
from kalshi_client import KalshiClient, OrderBook
from odds_client import BookQuote, OddsAPIClient, OddsEvent, OddsSnapshot

log = logging.getLogger("main")


# =============================================================================
# One scan
# =============================================================================
def run_scan(
    kalshi,
    fetch_snapshot: Callable[[], OddsSnapshot],
    *,
    journal: TradeJournal,
    risk: RiskManager,
    ledger: PaperLedger,
) -> dict:
    # 1. Ingest: Odds API snapshot (timestamps kept for staleness gates)
    snapshot = fetch_snapshot()
    print(f"Odds API: {len(snapshot.events)} games | credits remaining: {snapshot.credits_remaining} "
          f"| last call cost: {snapshot.last_call_cost}")

    # 2. Ingest: Kalshi markets (paginated), all configured NBA series
    kalshi_markets: list[dict] = []
    for series in config.KALSHI_SERIES_TICKERS:
        kalshi_markets.extend(kalshi.get_markets(series))
    print(f"Kalshi: {len(kalshi_markets)} open markets in {', '.join(config.KALSHI_SERIES_TICKERS)}")

    # 3. Account state: paper ledger in DRY_RUN, real balance/positions/orders when live
    now = datetime.now(timezone.utc)
    account: AccountState = ledger.account(now) if config.DRY_RUN else build_live_account(kalshi)
    print(f"Account ({account.source}): equity ${account.equity:.2f} | cash ${account.cash:.2f} "
          f"| exposure ${account.exposure:.2f}")

    # 4. Match games, then run the decision pipeline
    games, unmatched_odds, unmatched_kalshi = match_games(snapshot.events, kalshi_markets)
    engine = ExecutionEngine(kalshi, journal, risk, ledger)
    for event in unmatched_odds:
        engine.log_unmatched_odds(event, snapshot)
    for event_ticker in unmatched_kalshi:
        engine.log_unmatched_kalshi(event_ticker)
    engine.process_all(games, snapshot, account)

    # 5. Summary
    print("\n---- Scan summary ----")
    print(f"Matched games: {len(games)} | unmatched Odds API games: {len(unmatched_odds)} "
          f"| unmatched Kalshi events: {len(unmatched_kalshi)}")
    for decision, count in sorted(engine.counts.items()):
        print(f"  {decision:<16} {count}")
    if engine.buys:
        print(f"\n{'SIMULATED' if config.DRY_RUN else 'PLACED'} orders:")
        for buy in engine.buys:
            print(f"  BUY {buy['contracts']:>4} {buy['side'].upper():<3} {buy['ticker']} @ ${buy['price']:.2f} "
                  f"[{buy['route']}] net edge {buy['net_edge']:.2%} | max loss ${buy['stake']:.2f}")
    status = risk.last_status
    if status:
        print(f"\nRisk: day-start equity ${status['start_equity']:.2f} | drawdown {status['drawdown']:.2%} "
              f"(halt at {config.MAX_DAILY_DRAWDOWN:.0%}) | halted: {status['halted']}")
    print(f"Journal: {journal.csv_path}")
    return {"snapshot": snapshot, "engine": engine}


# =============================================================================
# Offline mock (synthetic Odds API + Kalshi data, exercises every decision path)
# =============================================================================
def _american_from_prob(prob: Decimal) -> Decimal:
    if prob >= Decimal("0.5"):
        return -((prob / (1 - prob)) * 100).to_integral_value()
    return (((1 - prob) / prob) * 100).to_integral_value()


def _quote(key: str, home: str, away: str, p_home: Decimal, updated: datetime) -> BookQuote:
    vig = Decimal("1.0225")  # ~4.5% two-way overround
    return BookQuote(
        book_key=key, book_title=key, last_update=updated,
        prices={home: _american_from_prob(p_home * vig), away: _american_from_prob((1 - p_home) * vig)},
    )


class MockKalshi:
    """Duck-typed stand-in for KalshiClient used by --mock. Never sends orders."""

    has_credentials = False

    def __init__(self) -> None:
        self.markets: list[dict] = []
        self.books: dict[str, OrderBook] = {}

    def add_game(self, event_ticker: str, away: str, home: str, bid: str, ask: str, ask_depth: int, bid_depth: int = 40):
        yes_bid, yes_ask = Decimal(bid), Decimal(ask)
        home_yes = ((yes_bid, Decimal(bid_depth)),)
        home_no = ((1 - yes_ask, Decimal(ask_depth)),)
        for code, yes_levels, no_levels in ((home, home_yes, home_no), (away, home_no, home_yes)):
            ticker = f"{event_ticker}-{code}"
            self.markets.append({"ticker": ticker, "event_ticker": event_ticker, "status": "open",
                                 "yes_sub_title": config.NBA_TEAMS[code][1]})
            self.books[ticker] = OrderBook(ticker, yes_levels, no_levels)

    def get_markets(self, series_ticker: str, status: str = "open") -> list[dict]:
        return list(self.markets)

    def get_orderbook(self, ticker: str) -> OrderBook:
        return self.books[ticker]

    def get_fee_schedule(self, series_ticker: str) -> me.FeeSchedule:
        return me.FeeSchedule("quadratic", Decimal("1"))

    def place_order(self, **_kwargs):  # pragma: no cover - guard
        raise RuntimeError("MockKalshi must never place orders.")


def build_mock_world() -> tuple[MockKalshi, Callable[[], OddsSnapshot]]:
    now = datetime.now(timezone.utc)
    kalshi = MockKalshi()
    events: list[OddsEvent] = []
    fresh = now - timedelta(seconds=15)
    stale = now - timedelta(minutes=10)

    # (away, home, p_home, commence_offset_min, book_update, kalshi quote or None, scenario)
    scenarios = [
        ("LAL", "HOU", "0.62", 240, fresh, ("0.53", "0.55", 40), "edge -> resting maker BUY"),
        ("BOS", "NYK", "0.70", 240, fresh, ("0.58", "0.60", 30), "big edge -> taker BUY"),
        ("MIA", "DEN", "0.55", 240, fresh, ("0.54", "0.56", 40), "fairly priced -> PASS"),
        ("CHI", "MIL", "0.60", 240, fresh, ("0.45", "0.53", 40), "wide spread -> SKIP_SPREAD"),
        ("DAL", "PHX", "0.62", 240, fresh, ("0.53", "0.55", 2), "thin ask depth -> SKIP_LIQUIDITY"),
        ("ATL", "BKN", "0.62", 240, stale, ("0.53", "0.55", 40), "stale odds -> SKIP_STALE"),
        ("TOR", "UTA", "0.58", 240, fresh, None, "no Kalshi market -> SKIP_NO_MATCH"),
        ("GSW", "SAC", "0.62", -30, fresh, ("0.53", "0.55", 40), "already started -> SKIP_LIVE"),
    ]
    for index, (away, home, p_home, offset_min, updated, kalshi_quote, _label) in enumerate(scenarios):
        commence = now + timedelta(minutes=offset_min)
        home_name = " ".join(config.NBA_TEAMS[home])
        away_name = " ".join(config.NBA_TEAMS[away])
        p = Decimal(p_home)
        quotes = [
            _quote("pinnacle", home_name, away_name, p, updated),
            _quote("lowvig", home_name, away_name, p + Decimal("0.004"), updated),
            _quote("draftkings", home_name, away_name, p - Decimal("0.008"), updated),
            _quote("fanduel", home_name, away_name, p - Decimal("0.005"), updated),
            _quote("betmgm", home_name, away_name, p + Decimal("0.006"), updated),
        ]
        events.append(OddsEvent(f"mock-{index}", home_name, away_name, commence, quotes))
        if kalshi_quote:
            date_code = commence.astimezone(config.GAME_TZ).strftime("%y%b%d").upper()
            kalshi.add_game(f"KXNBAGAME-{date_code}{away}{home}", away, home, *kalshi_quote)

    def fetch() -> OddsSnapshot:
        return OddsSnapshot(fetched_at=datetime.now(timezone.utc), events=events,
                            credits_remaining=480, credits_used=20, last_call_cost=1)

    return kalshi, fetch


# =============================================================================
# CLI
# =============================================================================
def print_banner(mock: bool) -> None:
    mode = "DRY_RUN (simulated orders)" if config.DRY_RUN else (
        "LIVE - ARMED" if config.live_trading_armed() else "LIVE requested but NOT armed")
    print("=" * 66)
    print(f" Kalshi NBA +EV engine | mode: {mode}{' | MOCK DATA' if mock else ''}")
    if not mock:
        print(f" Kalshi env: {config.KALSHI_ENV} ({config.KALSHI_BASE_URL})")
    print(f" devig: {config.DEVIG_METHOD} | min edge {config.MIN_NET_EDGE:.1%} | max spread ${config.MAX_SPREAD} "
          f"| Kelly x{config.KELLY_FRACTION}")
    print(f" caps: {config.MAX_BET_FRACTION:.0%} of equity / ${config.MAX_BET_DOLLARS} per bet | "
          f"daily stop -{config.MAX_DAILY_DRAWDOWN:.0%}")
    print("=" * 66)
    if config.TZ_FALLBACK_USED:
        print("WARNING: time-zone database not found (pip install tzdata); using fixed UTC offsets.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kalshi NBA +EV scanner / execution engine")
    parser.add_argument("--mock", action="store_true", help="run on synthetic offline data (forces no network)")
    parser.add_argument("--loop", action="store_true", help="rescan continuously")
    parser.add_argument("--interval", type=int, default=config.POLL_INTERVAL_SECONDS, help="seconds between scans")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print_banner(args.mock)

    if args.mock and not config.DRY_RUN:
        print("ERROR: --mock requires DRY_RUN = True.")
        return 2
    if not config.DRY_RUN and not config.live_trading_armed():
        print(f"ERROR: DRY_RUN is False but {config.LIVE_ARM_ENV_VAR} is not set to the arming phrase.\n"
              "       Refusing to start. Set DRY_RUN = True, or arm live trading deliberately.")
        return 2

    if args.mock:
        kalshi, fetch_snapshot = build_mock_world()
        data_dir = config.DATA_DIR
        journal_paths = (data_dir / "mock_trade_journal.csv", data_dir / "mock_trade_journal.jsonl")
        risk_path, ledger_path = data_dir / "mock_risk_state.json", data_dir / "mock_paper_ledger.json"
        for stale_file in (ledger_path, risk_path):  # every mock run starts from a clean sandbox
            stale_file.unlink(missing_ok=True)
    else:
        problems = config.validate_config()
        if problems:
            print("Configuration problems:\n  - " + "\n  - ".join(problems))
            return 2
        kalshi = KalshiClient()
        if not config.DRY_RUN and not kalshi.has_credentials:
            print("ERROR: live mode needs Kalshi credentials (KALSHI_KEY_ID + private key).")
            return 2
        if not kalshi.has_credentials:
            print("Note: no Kalshi credentials found; using public market data only (fine for DRY_RUN).")
        fetch_snapshot = OddsAPIClient().fetch_nba_odds
        journal_paths = (config.JOURNAL_CSV_PATH, config.JOURNAL_JSONL_PATH)
        risk_path, ledger_path = config.RISK_STATE_PATH, config.PAPER_LEDGER_PATH

    ledger = PaperLedger(ledger_path)
    risk = RiskManager(risk_path)
    try:
        while True:
            journal = TradeJournal(*journal_paths)
            result = run_scan(kalshi, fetch_snapshot, journal=journal, risk=risk, ledger=ledger)
            remaining = result["snapshot"].credits_remaining
            if not args.loop:
                break
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
