"""
test_engine.py - Offline test suite (no network, no real orders, no files outside a temp dir).

Run:  python -m unittest -v test_engine
"""
from __future__ import annotations

import base64
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from unittest import mock

import config
import execution_engine as ee
import main as main_mod
import math_engine as me
from kalshi_client import KalshiAPIError, KalshiAuthError, KalshiClient, parse_orderbook
from odds_client import OddsAPIClient, OddsEvent, OddsSnapshot


# ----------------------------------------------------------------------------- fakes
class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=None):
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})
        self.content = self.text.encode()
        self.headers = headers or {}

    def json(self, **kwargs):
        return json.loads(self.text, **kwargs)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, record):
        self.calls.append(record)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        return self._next({"method": method, "url": url, "params": params, "json": json, "headers": headers})

    def get(self, url, params=None, timeout=None):
        return self._next({"method": "GET", "url": url, "params": params})


def rsa_pem():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    return key, pem


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ----------------------------------------------------------------------------- math
class MathTests(unittest.TestCase):
    def test_devig_methods_sum_to_one(self):
        for odds in [(-110, -110), (-250, 210), (-1200, 800), (150, -170), (-105, -105)]:
            q = [me.american_to_implied_prob(o) for o in odds]
            for method in ("multiplicative", "power", "shin"):
                p = me.devig(q, method)
                self.assertAlmostEqual(float(sum(p)), 1.0, places=15, msg=f"{odds} {method}")
                self.assertTrue(all(0 < x < 1 for x in p))

    def test_power_and_shin_move_more_margin_off_the_longshot(self):
        q = [me.american_to_implied_prob(-1200), me.american_to_implied_prob(800)]
        mult, power, shin = (me.devig(q, m)[0] for m in ("multiplicative", "power", "shin"))
        self.assertGreater(power, mult)
        self.assertGreater(shin, mult)

    def test_no_vig_market_is_unchanged(self):
        p = me.devig([D("0.6"), D("0.4")], "power")
        self.assertEqual(round(p[0], 10), D("0.6"))

    def test_invalid_american_odds_rejected(self):
        for bad in (0, 50, -99):
            with self.assertRaises(ValueError):
                me.american_to_implied_prob(bad)

    def test_fee_formula_matches_kalshi_schedule(self):
        std = me.FeeSchedule("quadratic", D(1))
        with_maker = me.FeeSchedule("quadratic_with_maker_fees", D(1))
        self.assertEqual(me.fee_total(D("0.50"), 100, "taker", std), D("1.75"))
        self.assertEqual(me.fee_total(D("0.50"), 1, "taker", std), D("0.02"))  # rounds up to the cent
        self.assertEqual(me.fee_total(D("0.10"), 100, "taker", std), D("0.63"))
        self.assertEqual(me.fee_total(D("0.50"), 100, "maker", std), D("0"))
        self.assertEqual(me.fee_total(D("0.50"), 100, "maker", with_maker), D("0.44"))  # 0.4375 rounded up
        self.assertEqual(me.fee_total(D("0.50"), 100, "taker", me.FeeSchedule("quadratic", D("0.5"))), D("0.88"))

    def test_net_edge_formula(self):
        self.assertEqual(me.net_edge(D("0.60"), D("0.54"), D("0.01")), D("0.05"))

    def test_kelly_fraction(self):
        self.assertEqual(round(me.kelly_full_fraction(D("0.60"), D("0.53")), 6), D("0.148936"))
        self.assertEqual(me.kelly_full_fraction(D("0.50"), D("0.53")), D("0"))  # no edge -> no bet

    def test_sizing_is_quarter_kelly_and_capped(self):
        std = me.FeeSchedule("quadratic", D(1))
        small = me.size_order(fair_prob=D("0.60"), price=D("0.53"), role="maker", schedule=std,
                              equity=D(100), cash_available=D(100), exposure_room=D(30))
        self.assertEqual(small.binding_cap, "kelly")
        self.assertEqual(small.kelly_applied, small.kelly_full * D("0.25"))
        self.assertLessEqual(small.total_cost, small.budget)

        huge = me.size_order(fair_prob=D("0.95"), price=D("0.50"), role="taker", schedule=std,
                             equity=D(10000), cash_available=D(10000), exposure_room=D(3000))
        self.assertLessEqual(huge.total_cost, config.MAX_BET_DOLLARS)  # hard dollar cap
        self.assertEqual(huge.binding_cap, "max_bet_dollars")

        pct = me.size_order(fair_prob=D("0.95"), price=D("0.50"), role="maker", schedule=std,
                            equity=D(100), cash_available=D(100), exposure_room=D(30))
        self.assertLessEqual(pct.total_cost, D("5.00"))  # 5% of $100
        none = me.size_order(fair_prob=D("0.40"), price=D("0.50"), role="maker", schedule=std,
                             equity=D(100), cash_available=D(100), exposure_room=D(30))
        self.assertEqual(none.contracts, 0)

    def test_consensus_weights_sharp_books_more(self):
        home, away = "Houston Rockets", "Los Angeles Lakers"
        sharp = {home: D(-200), away: D(170)}
        soft = {home: D(-130), away: D(110)}
        weighted = me.build_consensus([("pinnacle", sharp), ("draftkings", soft)], (home, away), "power")
        flat = me.build_consensus([("fanduel", sharp), ("draftkings", soft)], (home, away), "power")
        self.assertGreater(weighted.probs[home], flat.probs[home])  # pulled toward the sharp price
        self.assertEqual(weighted.sharp_books_used, ("pinnacle",))
        self.assertEqual(weighted.probs[home] + weighted.probs[away], D(1))

    def test_no_float_arithmetic_in_financial_modules(self):
        for name in ("math_engine.py", "execution_engine.py", "kalshi_client.py"):
            source = (Path(__file__).parent / name).read_text(encoding="utf-8")
            self.assertNotIn("float(", source.replace("int(float(", ""), msg=name)


# ----------------------------------------------------------------------------- order book
class OrderBookTests(unittest.TestCase):
    DOC_EXAMPLE = {"orderbook_fp": {
        "yes_dollars": [["0.4100", "10.00"], ["0.4200", "13.00"], ["0.0100", "200.00"]],  # deliberately unsorted
        "no_dollars": [["0.5600", "17.00"], ["0.4400", "29.00"]],
    }}

    def test_complementary_asks_depth_and_spread(self):
        book = parse_orderbook("T", self.DOC_EXAMPLE)
        self.assertEqual(book.best_yes_bid, D("0.42"))
        self.assertEqual(book.best_no_bid, D("0.56"))
        self.assertEqual(book.best_ask("yes"), D("0.44"))  # 1 - best NO bid
        self.assertEqual(book.best_ask("no"), D("0.58"))  # 1 - best YES bid
        self.assertEqual(book.spread, D("0.02"))
        self.assertEqual(book.ask_depth("yes"), D("17"))  # buying YES consumes resting NO bids
        self.assertEqual(book.ask_depth("no"), D("13"))  # buying NO consumes resting YES bids
        self.assertEqual(book.ask_levels("yes")[0], (D("0.44"), D("17.00")))

    def test_legacy_cent_format_gives_same_book(self):
        legacy = {"orderbook": {"yes": [[42, 13]], "no": [[56, 17]]}}
        book = parse_orderbook("T", legacy)
        self.assertEqual((book.best_yes_bid, book.best_no_bid, book.spread), (D("0.42"), D("0.56"), D("0.02")))

    def test_empty_and_one_sided_books(self):
        empty = parse_orderbook("T", {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
        self.assertIsNone(empty.spread)
        self.assertEqual(empty.ask_depth("yes"), D(0))
        one_sided = parse_orderbook("T", {"orderbook_fp": {"yes_dollars": [["0.40", "5"]], "no_dollars": []}})
        self.assertIsNone(one_sided.best_ask("no") and one_sided.spread)


# ----------------------------------------------------------------------------- clients
class KalshiClientTests(unittest.TestCase):
    def make(self, responses, **kwargs):
        sleeps = []
        session = FakeSession(responses)
        client = KalshiClient(base_url="https://x.test/trade-api/v2", session=session,
                              sleep=sleeps.append, **kwargs)
        return client, session, sleeps

    def test_retries_429_and_5xx_with_exponential_backoff(self):
        client, session, sleeps = self.make([FakeResponse(429, text='{"error":"too many requests"}'),
                                             FakeResponse(503, text="down"), FakeResponse(200, {"ok": 1})])
        self.assertEqual(client._request("GET", "/markets"), {"ok": 1})
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(sleeps), 2)
        self.assertGreaterEqual(sleeps[0], 1.0)
        self.assertGreaterEqual(sleeps[1], 2.0)  # doubled

    def test_gives_up_after_max_retries(self):
        client, session, sleeps = self.make([FakeResponse(429, text="x")] * (config.HTTP_MAX_RETRIES + 1))
        with self.assertRaises(KalshiAPIError) as ctx:
            client._request("GET", "/markets")
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(len(sleeps), config.HTTP_MAX_RETRIES)

    def test_network_errors_are_retried_but_client_errors_are_not(self):
        import requests
        client, session, sleeps = self.make([requests.ConnectionError("boom"), FakeResponse(200, {"ok": 1})])
        self.assertEqual(client._request("GET", "/markets"), {"ok": 1})
        client, session, sleeps = self.make([FakeResponse(400, text="bad request")])
        with self.assertRaises(KalshiAPIError):
            client._request("GET", "/markets")
        self.assertEqual((len(session.calls), len(sleeps)), (1, 0))

    def test_cursor_pagination(self):
        client, session, _ = self.make([
            FakeResponse(200, {"markets": [{"ticker": "A"}], "cursor": "c1"}),
            FakeResponse(200, {"markets": [{"ticker": "B"}], "cursor": ""}),
        ])
        markets = client.get_markets("KXNBAGAME")
        self.assertEqual([m["ticker"] for m in markets], ["A", "B"])
        self.assertNotIn("cursor", session.calls[0]["params"])
        self.assertEqual(session.calls[1]["params"]["cursor"], "c1")

    def test_request_signature_verifies_and_excludes_query_string(self):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        key, pem = rsa_pem()
        client, session, _ = self.make([FakeResponse(200, {})], key_id="kid", private_key_pem=pem)
        client._request("GET", "/markets", params={"limit": 5}, require_auth=True)
        headers = session.calls[0]["headers"]
        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "kid")
        message = (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/markets").encode()
        key.public_key().verify(
            base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        signed = client._auth_headers("GET", "/markets?series_ticker=X&limit=5")
        key.public_key().verify(
            base64.b64decode(signed["KALSHI-ACCESS-SIGNATURE"]),
            (signed["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/markets").encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())

    def test_public_endpoints_are_sent_unsigned(self):
        _, pem = rsa_pem()
        client, session, _ = self.make([FakeResponse(200, {"markets": [], "cursor": ""}),
                                        FakeResponse(200, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}),
                                        FakeResponse(200, {"balance": 100, "portfolio_value": 0})],
                                       key_id="demo-key", private_key_pem=pem)
        client.get_markets("KXNBAGAME")
        client.get_orderbook("T-1")
        client.get_balance()
        self.assertNotIn("KALSHI-ACCESS-KEY", session.calls[0]["headers"])  # markets: public
        self.assertNotIn("KALSHI-ACCESS-KEY", session.calls[1]["headers"])  # order book: public
        self.assertEqual(session.calls[2]["headers"]["KALSHI-ACCESS-KEY"], "demo-key")  # balance: signed

    def test_account_endpoints_need_credentials(self):
        with mock.patch.object(config, "kalshi_key_id", return_value=None), \
                mock.patch.object(config, "kalshi_private_key_pem", return_value=None):
            client, _, _ = self.make([])
        with self.assertRaises(KalshiAuthError):
            client.get_balance()

    def test_place_order_payloads(self):
        _, pem = rsa_pem()
        client, session, _ = self.make([FakeResponse(201, {"order": {"order_id": "o1"}}),
                                        FakeResponse(201, {"order": {"order_id": "o2"}}),
                                        FakeResponse(409, text="duplicate client_order_id")],
                                       key_id="kid", private_key_pem=pem)
        client.place_order(ticker="T", side="yes", count=8, price=D("0.54"), client_order_id="c1", post_only=True)
        maker = session.calls[0]["json"]
        self.assertEqual((maker["side"], maker["action"], maker["type"], maker["count"], maker["yes_price"]),
                         ("yes", "buy", "limit", 8, 54))
        self.assertTrue(maker["post_only"])
        self.assertNotIn("time_in_force", maker)

        client.place_order(ticker="T", side="no", count=3, price=D("0.41"), client_order_id="c2",
                           post_only=False, fill_or_kill=True)
        taker = session.calls[1]["json"]
        self.assertEqual((taker["no_price"], taker["time_in_force"], taker["post_only"]), (41, "fill_or_kill", False))
        self.assertNotIn("yes_price", taker)

        dup = client.place_order(ticker="T", side="yes", count=1, price=D("0.5"), client_order_id="c1", post_only=True)
        self.assertTrue(dup["duplicate_client_order_id"])  # retry-after-timeout is idempotent
        with self.assertRaises(ValueError):
            client.place_order(ticker="T", side="yes", count=1, price=D("1.00"), client_order_id="c", post_only=True)

    def test_fee_schedule_comes_from_series_and_falls_back_conservatively(self):
        client, _, _ = self.make([FakeResponse(200, {"series": {"fee_type": "quadratic_with_maker_fees",
                                                                 "fee_multiplier": 0.5}})])
        schedule = client.get_fee_schedule("KXNBAGAME")
        self.assertEqual((schedule.fee_type, schedule.multiplier), ("quadratic_with_maker_fees", D("0.5")))
        client, _, _ = self.make([FakeResponse(404, text="nope")])
        self.assertEqual(client.get_fee_schedule("KXNBAGAME").fee_type, config.FALLBACK_FEE_TYPE)

    def test_balance_positions_and_orders_parsing(self):
        _, pem = rsa_pem()
        client, _, _ = self.make([
            FakeResponse(200, {"balance": 12345, "portfolio_value": 500}),
            FakeResponse(200, {"market_positions": [{"ticker": "A-1", "position_fp": "-10.00",
                                                     "market_exposure_dollars": "4.5000"},
                                                    {"ticker": "B-2", "position_fp": "0.00",
                                                     "market_exposure_dollars": "0.0000"}], "cursor": ""}),
            FakeResponse(200, {"orders": [{"ticker": "C-3", "order_id": "o", "side": "yes",
                                           "remaining_count": 5, "yes_price": 40}], "cursor": ""}),
        ], key_id="kid", private_key_pem=pem)
        self.assertEqual(client.get_balance(), {"cash": D("123.45"), "portfolio_value": D("5")})
        positions = client.get_positions()
        self.assertEqual([(p["ticker"], p["contracts"], p["exposure"]) for p in positions], [("A-1", D(10), D("4.5"))])
        resting = client.get_resting_orders()
        self.assertEqual((resting[0]["remaining"], resting[0]["price"], resting[0]["notional"]), (D(5), D("0.4"), D(2)))


class OddsClientTests(unittest.TestCase):
    PAYLOAD = [{"id": "e1", "commence_time": "2026-11-18T01:00:00Z", "home_team": "Houston Rockets",
                "away_team": "Los Angeles Lakers", "bookmakers": [
                    {"key": "pinnacle", "title": "Pinnacle", "last_update": "2026-11-17T23:00:00Z", "markets": [
                        {"key": "h2h", "last_update": "2026-11-17T23:00:05Z", "outcomes": [
                            {"name": "Houston Rockets", "price": -150}, {"name": "Los Angeles Lakers", "price": 130}]}]},
                    {"key": "badbook", "title": "Bad", "markets": [{"key": "h2h", "outcomes": [{"name": "x", "price": 1}]}]},
                ]}]
    HEADERS = {"x-requests-remaining": "480", "x-requests-used": "20", "x-requests-last": "1"}

    def test_parses_quotes_timestamps_and_credit_headers(self):
        session = FakeSession([FakeResponse(200, self.PAYLOAD, self.HEADERS)])
        snap = OddsAPIClient(api_key="k", session=session, sleep=lambda _s: None).fetch_nba_odds()
        self.assertEqual((snap.credits_remaining, snap.last_call_cost), (480, 1))
        quote = snap.events[0].quotes[0]
        self.assertEqual(len(snap.events[0].quotes), 1)  # malformed book dropped
        self.assertEqual(quote.last_update, datetime(2026, 11, 17, 23, 0, 5, tzinfo=timezone.utc))
        self.assertEqual(quote.prices["Houston Rockets"], D(-150))
        self.assertIsInstance(quote.prices["Houston Rockets"], D)
        self.assertEqual(session.calls[0]["params"]["bookmakers"], ",".join(config.ODDS_BOOKMAKERS))

    def test_retry_on_429_and_regions_fallback(self):
        sleeps = []
        session = FakeSession([FakeResponse(429, text="slow down"), FakeResponse(200, self.PAYLOAD, self.HEADERS)])
        OddsAPIClient(api_key="k", session=session, sleep=sleeps.append).fetch_nba_odds()
        self.assertEqual(len(sleeps), 1)
        session = FakeSession([FakeResponse(422, text="invalid bookmakers"), FakeResponse(200, self.PAYLOAD, self.HEADERS)])
        OddsAPIClient(api_key="k", session=session, sleep=lambda _s: None).fetch_nba_odds()
        self.assertIn("regions", session.calls[1]["params"])
        self.assertNotIn("bookmakers", session.calls[1]["params"])

    def test_book_staleness_split(self):
        event = OddsAPIClient._parse_event(self.PAYLOAD[0])
        now = datetime(2026, 11, 17, 23, 0, 40, tzinfo=timezone.utc)
        self.assertEqual(len(event.fresh_quotes(now, 60)[0]), 1)
        self.assertEqual(len(event.fresh_quotes(now + timedelta(seconds=30), 60)[1]), 1)  # 65s old -> stale


# ----------------------------------------------------------------------------- engine
class EngineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def build(self, kalshi, clock=None):
        journal = ee.TradeJournal(self.tmp / "j.csv", self.tmp / "j.jsonl", run_id="test")
        risk = ee.RiskManager(self.tmp / "risk.json", now_fn=clock)
        ledger = ee.PaperLedger(self.tmp / "ledger.json")
        return ee.ExecutionEngine(kalshi, journal, risk, ledger), journal, risk, ledger

    def jsonl(self):
        return [json.loads(line) for line in (self.tmp / "j.jsonl").read_text().splitlines()]

    def one_game(self, quotes_fn, *, snapshot_age=0, minutes_to_tip=240, kalshi_quote=("0.53", "0.55", 40)):
        now = datetime.now(timezone.utc)
        kalshi = main_mod.MockKalshi()
        home, away = "Houston Rockets", "Los Angeles Lakers"
        commence = now + timedelta(minutes=minutes_to_tip)
        event = OddsEvent("e1", home, away, commence, quotes_fn(home, away, now))
        code = commence.astimezone(config.GAME_TZ).strftime("%y%b%d").upper()
        kalshi.add_game(f"KXNBAGAME-{code}LALHOU", "LAL", "HOU", *kalshi_quote)
        snap = OddsSnapshot(fetched_at=now - timedelta(seconds=snapshot_age), events=[event], credits_remaining=100)
        return kalshi, snap

    def run_game(self, kalshi, snap, account=None):
        engine, journal, risk, ledger = self.build(kalshi)
        games, _, _ = ee.match_games(snap.events, kalshi.get_markets("KXNBAGAME"))
        account = account or ledger.account(datetime.now(timezone.utc))
        engine.process_all(games, snap, account)
        return engine

    @staticmethod
    def good_quotes(home, away, now, p="0.62"):
        fresh = now - timedelta(seconds=10)
        return [main_mod._quote(k, home, away, D(p), fresh) for k in ("pinnacle", "lowvig", "draftkings", "fanduel")]

    # -- matching
    def test_event_ticker_parsing(self):
        self.assertEqual(ee.parse_event_ticker("KXNBAGAME-26NOV17LALHOU"), (date(2026, 11, 17), "LAL", "HOU"))
        self.assertEqual(ee.parse_event_ticker("KXNBAGAME-26APR12ATLMIA"), (date(2026, 4, 12), "ATL", "MIA"))
        self.assertEqual(ee.parse_event_ticker("KXNBAGAME-26MAY04PHINYK")[1:], ("PHI", "NYK"))
        self.assertIsNone(ee.parse_event_ticker("KXNBA-26"))
        self.assertEqual(ee.odds_name_to_code("Portland Trail Blazers"), "POR")
        self.assertEqual(ee.odds_name_to_code("Los Angeles Clippers"), "LAC")

    def test_markets_map_to_the_right_team(self):
        now = datetime.now(timezone.utc)
        kalshi, snap = self.one_game(self.good_quotes)
        games, unmatched_odds, unmatched_kalshi = ee.match_games(snap.events, kalshi.get_markets("x"))
        self.assertEqual((len(games), unmatched_odds, unmatched_kalshi), (1, [], []))
        teams = {m["ticker"].rsplit("-", 1)[-1]: m["_yes_team"] for m in games[0].markets}
        self.assertEqual(teams, {"HOU": "Houston Rockets", "LAL": "Los Angeles Lakers"})

    # -- full mock scan + journal
    def test_full_mock_scan_journal_and_duplicate_prevention(self):
        ledger_path = self.tmp / "ledger.json"
        results = []
        for run in (1, 2):
            kalshi, fetch = main_mod.build_mock_world()
            journal = ee.TradeJournal(self.tmp / "j.csv", self.tmp / "j.jsonl", run_id=f"run{run}")
            result = quiet(main_mod.run_scan, kalshi, fetch, journal=journal,
                           risk=ee.RiskManager(self.tmp / "risk.json"), ledger=ee.PaperLedger(ledger_path))
            results.append(result["engine"])
        first, second = results
        self.assertEqual(first.counts["BUY"], 2)
        for decision in ("PASS", "SKIP_SPREAD", "SKIP_LIQUIDITY", "SKIP_STALE", "SKIP_LIVE", "SKIP_NO_MATCH"):
            self.assertGreater(first.counts[decision], 0, decision)
        self.assertEqual(second.counts["BUY"], 0)  # same games again: positions already on the (paper) books
        self.assertGreaterEqual(second.counts["SKIP_DUPLICATE"], 4)

        with (self.tmp / "j.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()), ee.JOURNAL_FIELDS)
        self.assertEqual(len(rows), len(self.jsonl()))
        self.assertEqual(len(rows), sum(first.counts.values()) + sum(second.counts.values()))
        buys = [r for r in self.jsonl() if r["decision"] == "BUY" and r["run_id"] == "run1"]
        for row in buys:
            self.assertGreaterEqual(D(row["net_edge"]), config.MIN_NET_EDGE)
            self.assertTrue(row["fair_prob_side"] and row["spread"] and row["client_order_id"])
            self.assertEqual(row["api_status"], "simulated")
        maker = next(r for r in buys if r["route"] == "maker")
        self.assertEqual(D(maker["order_price"]), D(maker["yes_bid"] if maker["side"] == "yes" else maker["no_bid"]) + D("0.01"))
        self.assertLess(D(maker["order_price"]), D(maker["yes_ask"] if maker["side"] == "yes" else maker["no_ask"]))
        liq = next(r for r in self.jsonl() if r["decision"] == "SKIP_LIQUIDITY")
        self.assertEqual(D(liq["ask_depth"]), D(2))
        self.assertTrue(all(r["decision"] in {d.value for d in ee.Decision} for r in self.jsonl()))

    # -- gates
    def test_gates_on_odds_quality(self):
        def decisions(engine):
            return set(engine.counts)

        def stale_soft(home, away, now):
            old = now - timedelta(minutes=10)
            fresh = now - timedelta(seconds=5)
            return [main_mod._quote("pinnacle", home, away, D("0.62"), fresh)] + [
                main_mod._quote(k, home, away, D("0.62"), old) for k in ("draftkings", "fanduel", "betmgm")]

        def no_sharp(home, away, now):
            fresh = now - timedelta(seconds=5)
            return [main_mod._quote(k, home, away, D("0.62"), fresh) for k in ("draftkings", "fanduel", "betmgm")]

        kalshi, snap = self.one_game(stale_soft)
        self.assertEqual(decisions(self.run_game(kalshi, snap)), {"SKIP_CONSENSUS"})  # 1 fresh book < 2 required
        kalshi, snap = self.one_game(no_sharp)
        self.assertEqual(decisions(self.run_game(kalshi, snap)), {"SKIP_CONSENSUS"})  # sharp book required
        kalshi, snap = self.one_game(self.good_quotes, snapshot_age=120)
        self.assertEqual(decisions(self.run_game(kalshi, snap)), {"SKIP_STALE"})  # snapshot older than 60s
        kalshi, snap = self.one_game(self.good_quotes, minutes_to_tip=2)
        self.assertEqual(decisions(self.run_game(kalshi, snap)), {"SKIP_LIVE"})
        kalshi, snap = self.one_game(self.good_quotes)
        self.assertEqual(self.run_game(kalshi, snap).counts["BUY"], 1)  # control: same setup trades when clean

    # -- risk controls
    def test_daily_drawdown_halt_is_sticky_and_resets_next_day(self):
        clock = [datetime(2026, 11, 17, 20, 0, tzinfo=timezone.utc)]
        risk = ee.RiskManager(self.tmp / "risk.json", now_fn=lambda: clock[0])

        def account(equity):
            return ee.AccountState(D(equity), D(equity), D(0), set(), set(), "paper")

        self.assertFalse(risk.check_halt(account(100)))
        self.assertFalse(risk.check_halt(account(86)))  # -14%: still trading
        self.assertTrue(risk.check_halt(account(84)))  # -16%: halted
        self.assertTrue(risk.check_halt(account(100)))  # recovery does not un-halt the same day
        clock[0] += timedelta(days=1)
        self.assertFalse(risk.check_halt(account(84)))  # new day, new baseline
        self.assertEqual(risk.last_status["start_equity"], D(84))

    def test_engine_places_nothing_when_halted(self):
        kalshi, snap = self.one_game(self.good_quotes)
        engine, _, risk, ledger = self.build(kalshi)
        risk.check_halt(ee.AccountState(D(100), D(100), D(0), set(), set(), "paper"))
        halted = ee.AccountState(D(80), D(80), D(0), set(), set(), "paper")
        games, _, _ = ee.match_games(snap.events, kalshi.get_markets("x"))
        engine.process_all(games, snap, halted)
        self.assertEqual(engine.counts["BUY"], 0)
        self.assertEqual(engine.counts["HALT_DRAWDOWN"], 2)

    def test_pre_trade_guardrails(self):
        risk = ee.RiskManager(self.tmp / "risk.json")

        def plan(cost, ticker="T-1"):
            return ee.OrderPlan(ticker=ticker, event_ticker="E-1", side="yes", price=D("0.5"), contracts=8,
                                route="maker", fee_total=D(0), total_cost=D(cost), net_edge=D("0.05"), fair_side=D("0.55"))

        def account(equity, cash, exposure=0, tickers=(), events=()):
            return ee.AccountState(D(equity), D(cash), D(exposure), set(tickers), set(events), "paper")

        self.assertEqual(risk.pre_trade_check(plan("4.00"), account(100, 100))[0], None)
        self.assertEqual(risk.pre_trade_check(plan("6.00"), account(100, 100))[0], ee.Decision.SKIP_RISK)  # >5%
        self.assertEqual(risk.pre_trade_check(plan("30"), account(10000, 10000))[0], ee.Decision.SKIP_RISK)  # >$25
        self.assertEqual(risk.pre_trade_check(plan("4.00"), account(100, 3))[0], ee.Decision.SKIP_RISK)  # no cash
        self.assertEqual(risk.pre_trade_check(plan("4.00"), account(100, 100, exposure=28))[0], ee.Decision.SKIP_RISK)
        self.assertEqual(risk.pre_trade_check(plan("4.00"), account(100, 100, tickers=["T-1"]))[0], ee.Decision.SKIP_DUPLICATE)
        self.assertEqual(risk.pre_trade_check(plan("4.00", "T-2"), account(100, 100, events=["E-1"]))[0], ee.Decision.SKIP_DUPLICATE)

    # -- live-trading interlock
    def test_live_orders_require_both_keys(self):
        for armed in (False, True):
            kalshi, snap = self.one_game(self.good_quotes)
            sent = []
            kalshi.place_order = lambda **kw: sent.append(kw) or {"order": {"order_id": "oid-1"}}
            env = {config.LIVE_ARM_ENV_VAR: config.LIVE_ARM_PHRASE} if armed else {}
            with mock.patch.object(config, "DRY_RUN", False), mock.patch.dict(os.environ, env):
                if not armed:
                    os.environ.pop(config.LIVE_ARM_ENV_VAR, None)
                live = ee.AccountState(D(100), D(100), D(0), set(), set(), "kalshi")
                engine = self.run_game(kalshi, snap, account=live)
            if armed:
                self.assertEqual(len(sent), 1)
                order = sent[0]
                self.assertEqual(order["post_only"], order["fill_or_kill"] is False)
                self.assertEqual(engine.counts["BUY"], 1)
                self.assertEqual(self.jsonl()[-1]["order_id"], "oid-1")
            else:
                self.assertEqual(sent, [])
                self.assertEqual(engine.counts["BUY"], 0)
                self.assertGreater(engine.counts["SKIP_RISK"], 0)

    def test_dry_run_never_calls_the_order_endpoint(self):
        kalshi, snap = self.one_game(self.good_quotes)
        kalshi.place_order = mock.Mock(side_effect=AssertionError("order endpoint must not be called in DRY_RUN"))
        self.assertTrue(config.DRY_RUN)
        self.assertEqual(self.run_game(kalshi, snap).counts["BUY"], 1)
        kalshi.place_order.assert_not_called()

    def test_main_refuses_to_start_live_without_arming(self):
        with mock.patch.object(config, "DRY_RUN", False), mock.patch.dict(os.environ):
            os.environ.pop(config.LIVE_ARM_ENV_VAR, None)
            self.assertEqual(quiet(main_mod.main, []), 2)
            self.assertEqual(quiet(main_mod.main, ["--mock"]), 2)

    def test_default_configuration_is_safe(self):
        self.assertIs(config.DRY_RUN, True)
        self.assertFalse(config.live_trading_armed())
        self.assertEqual(config.KELLY_FRACTION, D("0.25"))
        for value in (config.MIN_NET_EDGE, config.MAX_SPREAD, config.KELLY_FRACTION, config.MAX_DAILY_DRAWDOWN):
            self.assertIsInstance(value, D)


if __name__ == "__main__":
    unittest.main(verbosity=2)
