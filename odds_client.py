"""
odds_client.py - The Odds API client.

Endpoint strategy (this is what protects your monthly credit quota):

  * Moneylines (h2h), spreads and totals come from ONE bulk call:
        GET /v4/sports/basketball_nba/odds?markets=h2h,spreads,totals     cost = markets x regions
  * Player props are NOT available on the bulk endpoint. Only when "Player Props" is enabled do we
    use the per-event endpoint, once per game:
        GET /v4/sports/basketball_nba/events/{eventId}/odds?markets=player_points,...
                                                                          cost = markets x regions PER GAME
    and only for games the caller says Kalshi lists prop markets for (event_filter), that are
    pre-game, and that start within PROPS_MAX_HOURS_AHEAD.
  * If ONLY props are enabled, game ids come from the free GET /events endpoint.
  * Before every per-event call the client checks the credits left (from the response headers) and
    stops at MIN_ODDS_CREDITS_REMAINING, so a scan can never drain the quota.

Every bookmaker quote keeps its own last_update timestamp so the engine can enforce staleness gates.
Outcomes are normalised into SelectionKeys (team code / normalised player name / over-under + exact
point line) so the engine compares identical lines only. All prices are parsed into Decimal.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Iterable, Mapping, Set

import requests

import config
from math_engine import SelectionKey, normalize_player_name, team_code_from_text

log = logging.getLogger("odds_client")


class OddsAPIError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        # Fallback to current UTC time if upstream sends malformed timestamps to prevent crashing
        return datetime.now(timezone.utc)


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class BookQuote:
    """One bookmaker's quote for one Odds API market of one game."""

    book_key: str
    book_title: str
    last_update: datetime  # bookmaker-reported time of the quote (UTC)
    market: str  # Odds API market key: h2h | spreads | totals | player_points | ...
    selections: Mapping[SelectionKey, Decimal]  # canonical side -> American odds


@dataclass
class OddsEvent:
    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    quotes: list[BookQuote] = field(default_factory=list)
    players: dict[str, dict[str, str]] = field(default_factory=dict)  # odds market -> {normalised: display name}

    @property
    def home_code(self) -> str | None:
        return team_code_from_text(self.home_team)

    @property
    def away_code(self) -> str | None:
        return team_code_from_text(self.away_team)

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
    fetched_at: datetime  # when WE finished receiving the data (UTC)
    events: list[OddsEvent]
    credits_remaining: int | None = None
    credits_used: int | None = None
    last_call_cost: int | None = None
    scan_cost: int = 0  # credits this fetch consumed (sum of x-requests-last)
    api_calls: int = 0
    warnings: list[str] = field(default_factory=list)

    def age_seconds(self, now: datetime) -> float:
        return (now - self.fetched_at).total_seconds()


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def _decimal(value) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except ArithmeticError:
        return None


# =============================================================================
# Parsing: raw Odds API JSON -> canonical SelectionKeys
# =============================================================================
def parse_book_market(event: OddsEvent, book: dict, market: dict) -> BookQuote | None:
    """Normalise one bookmaker's market into SelectionKeys. Returns None if it is unusable."""
    key = market.get("key") or ""
    outcomes = market.get("outcomes") or []
    stamp = market.get("last_update") or book.get("last_update")
    if not stamp:
        return None
    try:
        updated = _parse_iso(stamp)
    except ValueError:
        return None

    selections: dict[SelectionKey, Decimal] = {}
    if key == "h2h":
        if len(outcomes) != 2:
            return None
        for outcome in outcomes:
            code, price = team_code_from_text(outcome.get("name")), _decimal(outcome.get("price"))
            if code is None or price is None:
                return None
            selections[SelectionKey("h2h", code, "win", None)] = price
    elif key == "spreads":
        if len(outcomes) != 2:
            return None
        for outcome in outcomes:
            code, price, point = team_code_from_text(outcome.get("name")), _decimal(outcome.get("price")), _decimal(outcome.get("point"))
            if code is None or price is None or point is None:
                return None
            selections[SelectionKey("spreads", code, "cover", point)] = price
    elif key == "totals":
        for outcome in outcomes:
            side, price, point = str(outcome.get("name", "")).lower(), _decimal(outcome.get("price")), _decimal(outcome.get("point"))
            if side in ("over", "under") and price is not None and point is not None:
                selections.setdefault(SelectionKey("totals", "game", side, point), price)
    elif key.startswith("player_"):
        for outcome in outcomes:
            # Fallback handling for books that flip name/description
            name_val = str(outcome.get("name", "")).lower()
            desc_val = str(outcome.get("description", ""))
            
            if name_val in ("over", "under"):
                side = name_val
                display = desc_val
            else:
                side = desc_val.lower()
                display = name_val
                
            player = normalize_player_name(display)
            price = _decimal(outcome.get("price"))
            point = _decimal(outcome.get("point"))
            
            if side in ("over", "under") and player and price is not None and point is not None:
                selections.setdefault(SelectionKey(key, player, side, point), price)
                event.players.setdefault(key, {})[player] = display
    else:
        return None

    if len(selections) < 2:
        return None
    return BookQuote(
        book_key=book.get("key", ""),
        book_title=book.get("title", book.get("key", "")),
        last_update=updated,
        market=key,
        selections=selections,
    )


def add_quotes(event: OddsEvent, raw: dict) -> int:
    """Parse every bookmaker market in a raw event payload into `event.quotes`. Returns quotes added."""
    added = 0
    for book in raw.get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            quote = parse_book_market(event, book, market)
            if quote is not None:
                event.quotes.append(quote)
                added += 1
    return added


def parse_event(raw: dict) -> OddsEvent | None:
    try:
        event = OddsEvent(
            event_id=str(raw["id"]),
            home_team=raw["home_team"],
            away_team=raw["away_team"],
            commence_time=_parse_iso(raw["commence_time"]),
        )
    except (KeyError, ValueError):
        return None
    add_quotes(event, raw)
    return event


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
        self._use_regions = not config.ODDS_BOOKMAKERS
        self.credits_remaining: int | None = None
        self.credits_used: int | None = None
        self.last_call_cost: int | None = None
        self._scan_cost = 0
        self._api_calls = 0
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
            self._note_usage(response)
            return response

    def _note_usage(self, response: requests.Response) -> None:
        self._api_calls += 1
        remaining = _header_int(response.headers, "x-requests-remaining")
        if remaining is not None:
            self.credits_remaining = remaining
        used = _header_int(response.headers, "x-requests-used")
        if used is not None:
            self.credits_used = used
        last = _header_int(response.headers, "x-requests-last")
        if last is not None:
            self.last_call_cost = last
            self._scan_cost += last

    def _base_params(self, markets: str | None = None) -> dict:
        params = {"apiKey": self.api_key, "oddsFormat": "american", "dateFormat": "iso"}
        if markets:
            params["markets"] = markets
        if self._use_regions:
            params["regions"] = config.ODDS_REGIONS
        else:
            params["bookmakers"] = ",".join(config.ODDS_BOOKMAKERS)
        return params

    @staticmethod
    def _raise_for_status(response: requests.Response, what: str) -> None:
        if response.status_code == 401:
            raise OddsAPIError(f"Odds API auth/quota error (401) on {what}: {response.text[:200]}", 401)
        if response.status_code != 200:
            raise OddsAPIError(f"Odds API HTTP {response.status_code} on {what}: {response.text[:200]}", response.status_code)

    # ------------------------------------------------------------------ public
    def fetch_nba_odds(
        self,
        categories: Iterable[str] | None = None,
        event_filter: Callable[[OddsEvent], bool] | None = None,
        target_event_ids: Set[str] | None = None,
    ) -> OddsSnapshot:
        """
        Fetch NBA odds for the enabled market categories.

        categories       subset of {"moneyline","spread","total","prop"}; default = config.enabled_categories().
        event_filter     optional predicate limiting WHICH games get the (expensive) per-event prop call.
        target_event_ids optional set of specific explicit game IDs to pull props for.
        """
        if not self.api_key:
            raise OddsAPIError("ODDS_API_KEY is not set.")
        categories = tuple(categories) if categories is not None else config.enabled_categories()
        bulk_markets = config.bulk_odds_markets(categories)
        prop_markets = config.prop_odds_markets(categories)
        if not bulk_markets and not prop_markets:
            raise OddsAPIError("No market types selected.")

        self._scan_cost, self._api_calls = 0, 0
        warnings: list[str] = []
        now = datetime.now(timezone.utc)
        horizon = _iso_z(now + timedelta(hours=config.ODDS_MAX_HOURS_AHEAD))
        sport_url = f"{config.ODDS_BASE_URL}/sports/{config.ODDS_SPORT}"

        # ---- Step 1: one bulk call for h2h/spreads/totals, or the FREE /events list for props-only
        if bulk_markets:
            url = f"{sport_url}/odds"
            params = {**self._base_params(",".join(bulk_markets)), "commenceTimeTo": horizon}
            response = self._get(url, params)
            if not self._use_regions and response.status_code in (400, 422):
                log.warning("Odds API rejected the bookmakers list (%s); falling back to regions=%s",
                            response.text[:120], config.ODDS_REGIONS)
                self._use_regions = True
                params = {**self._base_params(",".join(bulk_markets)), "commenceTimeTo": horizon}
                response = self._get(url, params)
            self._raise_for_status(response, "bulk /odds")
            payload = response.json(parse_float=Decimal)
        else:
            response = self._get(f"{sport_url}/events", {"apiKey": self.api_key, "dateFormat": "iso",
                                                         "commenceTimeTo": horizon})
            self._raise_for_status(response, "/events")
            payload = response.json(parse_float=Decimal)
        events = [e for e in (parse_event(raw) for raw in payload) if e is not None]

        # Filter bulk events by explicitly requested event IDs if provided
        if target_event_ids is not None:
            events = [e for e in events if e.event_id in target_event_ids]

        # ---- Step 2: per-event prop calls (only when props are enabled)
        props_fetched = 0
        if prop_markets:
            regions = config.odds_regions_count()
            cost_each = len(prop_markets) * regions
            cutoff = now + timedelta(hours=config.PROPS_MAX_HOURS_AHEAD)
            earliest = now + timedelta(minutes=config.MIN_MINUTES_BEFORE_TIPOFF)
            candidates = sorted(
                (e for e in events
                 if earliest < e.commence_time <= cutoff and (event_filter is None or event_filter(e))),
                key=lambda e: e.commence_time,
            )
            skipped = 0
            for index, event in enumerate(candidates):
                left = self.credits_remaining
                # Fix: Failsafe assumption if API removes x-requests header
                if left is None or left - cost_each < config.MIN_ODDS_CREDITS_REMAINING:
                    skipped = len(candidates) - index
                    left_display = left if left is not None else "Unknown"
                    warnings.append(
                        f"Stopped fetching player props: {left_display} credits left (floor "
                        f"{config.MIN_ODDS_CREDITS_REMAINING}); {skipped} game(s) skipped."
                    )
                    break
                url = f"{sport_url}/events/{event.event_id}/odds"
                response = self._get(url, self._base_params(",".join(prop_markets)))
                if response.status_code in (404, 422):
                    warnings.append(f"No props for {event.away_team} @ {event.home_team} "
                                    f"(HTTP {response.status_code}).")
                    continue
                self._raise_for_status(response, f"event odds {event.event_id}")
                raw = response.json(parse_float=Decimal)
                if isinstance(raw, dict):
                    add_quotes(event, raw)
                    props_fetched += 1
            if not candidates:
                warnings.append("Player props enabled, but no upcoming game with Kalshi prop markets was found.")

        snapshot = OddsSnapshot(
            fetched_at=datetime.now(timezone.utc),
            events=events,
            credits_remaining=self.credits_remaining,
            credits_used=self.credits_used,
            last_call_cost=self.last_call_cost,
            scan_cost=self._scan_cost,
            api_calls=self._api_calls,
            warnings=warnings,
        )
        log.info("Fetched odds for %d games (%d prop calls, %d credits this scan, %s remaining)",
                 len(events), props_fetched, self._scan_cost, self.credits_remaining)
        return snapshot
