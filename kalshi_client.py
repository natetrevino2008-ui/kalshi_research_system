"""
kalshi_client.py - Kalshi Trade API client.

* RSA-PSS request signing via the `cryptography` package (signs timestamp + METHOD +
  path, with the query string stripped, as Kalshi requires).
* Public endpoints (markets, order books, series) work without credentials, so the
  scanner can run in DRY_RUN with no keys. Account endpoints require credentials.
* Exponential backoff with jitter on HTTP 429, 5xx and network errors. Kalshi's 429
  carries no Retry-After header, so backoff is the recommended strategy.
* Cursor pagination for markets, positions and orders.
* Order book parsing for both the current `orderbook_fp` format (string dollars) and the
  legacy `orderbook` format (integer cents). Books contain BIDS ONLY: a YES ask is
  1 - best NO bid, and buying YES consumes resting NO bids (and vice versa).
"""
from __future__ import annotations

import base64
import logging
import random
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterator
from urllib.parse import urlparse

import requests

import config
from math_engine import FeeSchedule, ONE, ZERO, to_decimal

log = logging.getLogger("kalshi_client")


class KalshiAPIError(Exception):
    def __init__(self, status: int, body: str, endpoint: str = ""):
        super().__init__(f"Kalshi HTTP {status} on {endpoint}: {body[:300]}")
        self.status = status
        self.body = body
        self.endpoint = endpoint


class KalshiAuthError(Exception):
    pass


# =============================================================================
# Order book
# =============================================================================
@dataclass(frozen=True)
class OrderBook:
    """Resting bids only, each side sorted by price ascending (best bid is last)."""

    ticker: str
    yes_bids: tuple[tuple[Decimal, Decimal], ...]  # (price in dollars, contracts)
    no_bids: tuple[tuple[Decimal, Decimal], ...]

    @property
    def best_yes_bid(self) -> Decimal | None:
        return self.yes_bids[-1][0] if self.yes_bids else None

    @property
    def best_no_bid(self) -> Decimal | None:
        return self.no_bids[-1][0] if self.no_bids else None

    def best_bid(self, side: str) -> Decimal | None:
        return self.best_yes_bid if side == "yes" else self.best_no_bid

    def best_ask(self, side: str) -> Decimal | None:
        """Buying YES lifts resting NO bids: YES ask = 1 - best NO bid (and vice versa)."""
        opposite = self.best_no_bid if side == "yes" else self.best_yes_bid
        return None if opposite is None else ONE - opposite

    def ask_depth(self, side: str) -> Decimal:
        """Contracts available to BUY `side` at the best ask = size of the best opposite bid."""
        opposite = self.no_bids if side == "yes" else self.yes_bids
        return opposite[-1][1] if opposite else ZERO

    def ask_levels(self, side: str) -> list[tuple[Decimal, Decimal]]:
        """Ask ladder for buying `side`, cheapest first: [(ask_price, contracts), ...]."""
        opposite = self.no_bids if side == "yes" else self.yes_bids
        return [(ONE - price, count) for price, count in reversed(opposite)]

    @property
    def spread(self) -> Decimal | None:
        """YES spread == NO spread == 1 - best_yes_bid - best_no_bid."""
        if self.best_yes_bid is None or self.best_no_bid is None:
            return None
        return ONE - self.best_yes_bid - self.best_no_bid


def _levels(raw_levels, cents: bool) -> tuple[tuple[Decimal, Decimal], ...]:
    levels: list[tuple[Decimal, Decimal]] = []
    for level in raw_levels or []:
        try:
            price = to_decimal(level[0])
            count = to_decimal(level[1])
        except (IndexError, TypeError, ArithmeticError):
            continue
        if cents:
            price = price / 100
        if count > 0 and 0 < price < 1:
            levels.append((price, count))
    levels.sort(key=lambda lv: lv[0])
    return tuple(levels)


def parse_orderbook(ticker: str, payload: dict) -> OrderBook:
    """Accepts {'orderbook_fp': {yes_dollars, no_dollars}} or legacy {'orderbook': {yes, no}}."""
    fp = payload.get("orderbook_fp")
    if fp:
        return OrderBook(
            ticker=ticker,
            yes_bids=_levels(fp.get("yes_dollars"), cents=False),
            no_bids=_levels(fp.get("no_dollars"), cents=False),
        )
    legacy = payload.get("orderbook") or {}
    return OrderBook(
        ticker=ticker,
        yes_bids=_levels(legacy.get("yes_dollars") or legacy.get("yes"), cents=not legacy.get("yes_dollars")),
        no_bids=_levels(legacy.get("no_dollars") or legacy.get("no"), cents=not legacy.get("no_dollars")),
    )


def _dec_field(record: dict, *names: str, default: Decimal = ZERO) -> Decimal:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            try:
                return to_decimal(value)
            except ArithmeticError:
                continue
    return default


# =============================================================================
# Client
# =============================================================================
class KalshiClient:
    def __init__(
        self,
        base_url: str | None = None,
        key_id: str | None = None,
        private_key_pem: bytes | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = (base_url or config.KALSHI_BASE_URL).rstrip("/")
        self._path_prefix = urlparse(self.base_url).path  # e.g. /trade-api/v2
        self.key_id = key_id or config.kalshi_key_id()
        pem = private_key_pem or config.kalshi_private_key_pem()
        self._private_key = self._load_private_key(pem) if (self.key_id and pem) else None
        self.session = session or requests.Session()
        self._sleep = sleep
        self._fee_cache: dict[str, FeeSchedule] = {}

    # ------------------------------------------------------------------ auth
    @staticmethod
    def _load_private_key(pem: bytes):
        from cryptography.hazmat.primitives import serialization

        return serialization.load_pem_private_key(pem, password=None)

    @property
    def has_credentials(self) -> bool:
        return self._private_key is not None

    def _auth_headers(self, method: str, endpoint: str) -> dict[str, str]:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp = str(int(time.time() * 1000))
        path = self._path_prefix + endpoint.split("?")[0]  # query string is NOT signed
        message = f"{timestamp}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    # ------------------------------------------------------------------ HTTP core
    def _backoff(self, attempt: int) -> float:
        delay = min(config.BACKOFF_MAX_SECONDS, config.BACKOFF_BASE_SECONDS * (2**attempt))
        return delay + random.uniform(0, delay * 0.25)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        require_auth: bool = False,
    ) -> dict:
        if require_auth and not self.has_credentials:
            raise KalshiAuthError(
                f"{method} {endpoint} needs API credentials (KALSHI_KEY_ID + private key)."
            )
        url = self.base_url + endpoint
        attempt = 0
        while True:
            headers = {"Accept": "application/json"}
            if json_body is not None:
                headers["Content-Type"] = "application/json"
            if require_auth and self.has_credentials:
                # Only account/order endpoints are signed (fresh timestamp per attempt). Public market
                # data is sent unsigned so stale or wrong-environment keys can never break a scan.
                headers.update(self._auth_headers(method, endpoint))
            try:
                response = self.session.request(
                    method, url, params=params, json=json_body, headers=headers,
                    timeout=config.HTTP_TIMEOUT_SECONDS,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= config.HTTP_MAX_RETRIES:
                    raise KalshiAPIError(0, f"network error after {attempt} retries: {exc}", endpoint) from exc
                delay = self._backoff(attempt)
                log.warning("Kalshi network error on %s (%s); retry %d in %.1fs", endpoint, exc, attempt + 1, delay)
                self._sleep(delay)
                attempt += 1
                continue

            status = response.status_code
            if status == 429 or status >= 500:
                if attempt >= config.HTTP_MAX_RETRIES:
                    raise KalshiAPIError(status, response.text, endpoint)
                delay = self._backoff(attempt)
                log.warning("Kalshi HTTP %d on %s; retry %d in %.1fs", status, endpoint, attempt + 1, delay)
                self._sleep(delay)
                attempt += 1
                continue
            if 200 <= status < 300:
                return response.json(parse_float=Decimal) if response.content else {}
            raise KalshiAPIError(status, response.text, endpoint)

    def _paginate(self, endpoint: str, list_key: str, params: dict | None = None, require_auth: bool = False) -> Iterator[dict]:
        cursor = ""
        for _ in range(config.KALSHI_MAX_PAGES):
            page_params = dict(params or {})
            page_params["limit"] = config.KALSHI_MARKETS_PAGE_LIMIT
            if cursor:
                page_params["cursor"] = cursor
            payload = self._request("GET", endpoint, params=page_params, require_auth=require_auth)
            for item in payload.get(list_key) or []:
                yield item
            cursor = payload.get("cursor") or ""
            if not cursor:
                return
        log.warning("Pagination for %s stopped at KALSHI_MAX_PAGES=%d", endpoint, config.KALSHI_MAX_PAGES)

    # ------------------------------------------------------------------ market data (public)
    def get_markets(self, series_ticker: str, status: str = "open") -> list[dict]:
        return list(self._paginate("/markets", "markets", {"series_ticker": series_ticker, "status": status}))

    def get_orderbook(self, ticker: str) -> OrderBook:
        return parse_orderbook(ticker, self._request("GET", f"/markets/{ticker}/orderbook"))

    def get_fee_schedule(self, series_ticker: str) -> FeeSchedule:
        """Series-level fee_type / fee_multiplier. Falls back conservatively if the lookup fails."""
        if series_ticker in self._fee_cache:
            return self._fee_cache[series_ticker]
        try:
            series = self._request("GET", f"/series/{series_ticker}").get("series") or {}
            schedule = FeeSchedule(
                fee_type=str(series.get("fee_type") or "quadratic"),
                multiplier=to_decimal(series.get("fee_multiplier", 1)),
            )
        except (KalshiAPIError, ArithmeticError) as exc:
            log.warning("Series lookup failed for %s (%s); assuming maker fees apply.", series_ticker, exc)
            schedule = FeeSchedule(config.FALLBACK_FEE_TYPE, config.FALLBACK_FEE_MULTIPLIER)
        self._fee_cache[series_ticker] = schedule
        return schedule

    # ------------------------------------------------------------------ account (authenticated)
    def get_balance(self) -> dict[str, Decimal]:
        """Returns {'cash': available balance, 'portfolio_value': value of positions} in dollars."""
        payload = self._request("GET", "/portfolio/balance", require_auth=True)
        if "balance_dollars" in payload:
            cash = _dec_field(payload, "balance_dollars")
        else:
            cash = _dec_field(payload, "balance") / 100
        if "portfolio_value_dollars" in payload:
            positions_value = _dec_field(payload, "portfolio_value_dollars")
        else:
            positions_value = _dec_field(payload, "portfolio_value") / 100
        return {"cash": cash, "portfolio_value": positions_value}

    def get_positions(self) -> list[dict]:
        """Open positions as [{'ticker', 'contracts' (abs), 'exposure' (dollars)}]."""
        out: list[dict] = []
        for pos in self._paginate("/portfolio/positions", "market_positions", require_auth=True):
            contracts = abs(_dec_field(pos, "position_fp", "position"))
            if "market_exposure_dollars" in pos:
                exposure = _dec_field(pos, "market_exposure_dollars")
            else:
                exposure = _dec_field(pos, "market_exposure") / 100
            if contracts > 0 or exposure > 0:
                out.append({"ticker": pos.get("ticker", ""), "contracts": contracts, "exposure": exposure})
        return out

    def get_resting_orders(self) -> list[dict]:
        """Resting orders as [{'ticker', 'order_id', 'side', 'remaining', 'price', 'notional'}]."""
        out: list[dict] = []
        for order in self._paginate("/portfolio/orders", "orders", {"status": "resting"}, require_auth=True):
            side = order.get("side", "yes")
            remaining = _dec_field(order, "remaining_count_fp", "remaining_count")
            if f"{side}_price_dollars" in order:
                price = _dec_field(order, f"{side}_price_dollars")
            else:
                price = _dec_field(order, f"{side}_price") / 100
            out.append({
                "ticker": order.get("ticker", ""),
                "order_id": order.get("order_id", ""),
                "side": side,
                "remaining": remaining,
                "price": price,
                "notional": remaining * price,
            })
        return out

    # ------------------------------------------------------------------ orders
    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        count: int,
        price: Decimal,
        client_order_id: str,
        post_only: bool,
        fill_or_kill: bool = False,
    ) -> dict:
        """
        Buy `count` contracts of `side` ('yes'/'no') with a limit order at `price` dollars.
        Maker orders pass post_only=True (rejected rather than crossing the spread).
        Taker orders pass fill_or_kill=True (all-or-nothing at the limit price).
        A fixed client_order_id makes retries after a network failure safe; a 409 means the
        order was already accepted.
        """
        if side not in ("yes", "no"):
            raise ValueError("side must be 'yes' or 'no'")
        cents = int((price * 100).to_integral_value())
        if not 1 <= cents <= 99:
            raise ValueError(f"price {price} outside 0.01-0.99")
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "action": "buy",
            "count": int(count),
            "type": "limit",
            f"{side}_price": cents,
            "post_only": bool(post_only),
        }
        if fill_or_kill:
            body["time_in_force"] = "fill_or_kill"
        try:
            return self._request("POST", "/portfolio/orders", json_body=body, require_auth=True)
        except KalshiAPIError as exc:
            if exc.status == 409:
                return {"duplicate_client_order_id": True, "client_order_id": client_order_id, "error": exc.body}
            raise

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/orders/{order_id}", require_auth=True)
