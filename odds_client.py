"""
odds_client.py - The Odds API client.

Fetches NBA moneylines (h2h) from many bookmakers, keeps every bookmaker's own
last_update timestamp so the engine can enforce staleness gates, and retries on
HTTP 429 / 5xx / network errors with exponential backoff + jitter.

All prices are parsed straight into Decimal (json parse_float=Decimal).
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Mapping

import requests

import config

log = logging.getLogger("odds_client")


class OddsAPIError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class BookQuote:
    book_key: str
    book_title: str
    last_update: datetime  # bookmaker-reported time of the quote (UTC)
    prices: Mapping[str, Decimal]  # team name -> American odds


@dataclass
class OddsEvent:
    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    quotes: list[BookQuote] = field(default_factory=list)

    def fresh_quotes(self, now: datetime, max_age_seconds: int) -> tuple[list[BookQuote], list[BookQuote]]:
        """Split quotes into (fresh, stale) by the bookmaker's own last_update timestamp."""
        fresh: list[BookQuote] = []
        stale: list[BookQuote] = []
        for quote in self.quotes:
            age = (now - quote.last_update).total_seconds()
            (fresh if age <= max_age_seconds else stale).append(quote)
        return fresh, stale


@dataclass
class OddsSnapshot:
    fetched_at: datetime  # when WE received the response (UTC)
    events: list[OddsEvent]
    credits_remaining: int | None = None
    credits_used: int | None = None
    last_call_cost: int | None = None

    def age_seconds(self, now: datetime) -> float:
        return (now - self.fetched_at).total_seconds()


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


class OddsAPIClient:
    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key or config.ODDS_API_KEY
        self.session = session or requests.Session()
        self._sleep = sleep
        if not self.api_key:
            log.warning("ODDS_API_KEY is missing; fetch_nba_odds() will raise.")

    # ------------------------------------------------------------------ HTTP
    def _backoff(self, attempt: int) -> float:
        delay = min(config.BACKOFF_MAX_SECONDS, config.BACKOFF_BASE_SECONDS * (2**attempt))
        return delay + random.uniform(0, delay * 0.25)

    def _get(self, url: str, params: dict) -> requests.Response:
        attempt = 0
        while True:
            try:
                response = self.session.get(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= config.HTTP_MAX_RETRIES:
                    raise OddsAPIError(f"Network error after {attempt} retries: {exc}") from exc
                delay = self._backoff(attempt)
                log.warning("Odds API network error (%s); retry %d in %.1fs", exc, attempt + 1, delay)
                self._sleep(delay)
                attempt += 1
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= config.HTTP_MAX_RETRIES:
                    raise OddsAPIError(
                        f"HTTP {response.status_code} after {attempt} retries: {response.text[:200]}",
                        response.status_code,
                    )
                delay = self._backoff(attempt)
                log.warning("Odds API HTTP %d; retry %d in %.1fs", response.status_code, attempt + 1, delay)
                self._sleep(delay)
                attempt += 1
                continue
            return response

    # ------------------------------------------------------------------ public
    def fetch_nba_odds(self) -> OddsSnapshot:
        """Fetch current NBA moneylines from every configured bookmaker."""
        if not self.api_key:
            raise OddsAPIError("ODDS_API_KEY is not set.")

        url = f"{config.ODDS_BASE_URL}/sports/{config.ODDS_SPORT}/odds"
        base_params = {
            "apiKey": self.api_key,
            "markets": config.ODDS_MARKETS,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        params = dict(base_params)
        using_bookmakers = bool(config.ODDS_BOOKMAKERS)
        if using_bookmakers:
            params["bookmakers"] = ",".join(config.ODDS_BOOKMAKERS)
        else:
            params["regions"] = config.ODDS_REGIONS

        response = self._get(url, params)
        if using_bookmakers and response.status_code in (400, 422):
            log.warning("Odds API rejected the bookmakers list (%s); falling back to regions=%s",
                        response.text[:120], config.ODDS_REGIONS)
            params = dict(base_params, regions=config.ODDS_REGIONS)
            response = self._get(url, params)

        fetched_at = datetime.now(timezone.utc)
        if response.status_code == 401:
            raise OddsAPIError(f"Odds API auth/quota error (401): {response.text[:200]}", 401)
        if response.status_code != 200:
            raise OddsAPIError(f"Odds API HTTP {response.status_code}: {response.text[:200]}", response.status_code)

        payload = response.json(parse_float=Decimal)
        events = [self._parse_event(raw) for raw in payload]
        snapshot = OddsSnapshot(
            fetched_at=fetched_at,
            events=[e for e in events if e is not None],
            credits_remaining=_header_int(response.headers, "x-requests-remaining"),
            credits_used=_header_int(response.headers, "x-requests-used"),
            last_call_cost=_header_int(response.headers, "x-requests-last"),
        )
        log.info("Fetched odds for %d games (credits remaining: %s, last call cost: %s)",
                 len(snapshot.events), snapshot.credits_remaining, snapshot.last_call_cost)
        return snapshot

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse_event(raw: dict) -> OddsEvent | None:
        try:
            event = OddsEvent(
                event_id=str(raw["id"]),
                home_team=raw["home_team"],
                away_team=raw["away_team"],
                commence_time=_parse_iso(raw["commence_time"]),
            )
        except (KeyError, ValueError):
            return None

        for book in raw.get("bookmakers", []) or []:
            for market in book.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes") or []
                if len(outcomes) != 2:
                    continue
                stamp = market.get("last_update") or book.get("last_update")
                if not stamp:
                    continue
                try:
                    prices = {o["name"]: Decimal(str(o["price"])) for o in outcomes}
                    event.quotes.append(
                        BookQuote(
                            book_key=book.get("key", ""),
                            book_title=book.get("title", book.get("key", "")),
                            last_update=_parse_iso(stamp),
                            prices=prices,
                        )
                    )
                except (KeyError, ValueError, ArithmeticError):
                    continue
        return event
