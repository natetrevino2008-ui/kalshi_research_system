"""
math_engine.py - Pure pricing / fee / sizing math. Everything is decimal.Decimal.

No network, no file I/O, no clocks: every function is deterministic so it can be unit
tested offline.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Iterable, Mapping, NamedTuple, Sequence

import config

ZERO = Decimal("0")
ONE = Decimal("1")
CENT = Decimal("0.01")
_EPS = Decimal("1e-20")


def to_decimal(value) -> Decimal:
    """Convert int / str / float / Decimal to Decimal without binary-float artefacts."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def ceil_to_cent(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_CEILING)


# =============================================================================
# 0. Name and line normalisation
# =============================================================================
class SelectionKey(NamedTuple):
    market: str
    subject: str
    side: str
    line: Decimal | None


def canonical_team_code(code: str) -> str | None:
    code = code.upper()
    code = config.NBA_TEAM_CODE_ALIASES.get(code, code)
    return code if code in config.NBA_TEAMS else None


def team_code_from_text(text: str | None) -> str | None:
    if not text:
        return None
    low = re.sub(r"\s+", " ", text.lower()).strip()
    for alias, code in config.TEAM_TEXT_ALIASES.items():
        if low.startswith(alias):
            return code
    for code, (_city, nickname) in config.NBA_TEAMS.items():
        if re.search(r"\b" + re.escape(nickname.lower()) + r"\b", low):
            return code
    matches = [
        code for code, (city, _nick) in config.NBA_TEAMS.items()
        if city != "Los Angeles" and low.startswith(city.lower())
    ]
    return matches[0] if len(matches) == 1 else None


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize_player_name(name: str | None) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[.'\u2019`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return " ".join(part for part in text.split() if part not in _NAME_SUFFIXES)


def players_match(a: str, b: str) -> bool:
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    if len(ta) < 2 or len(tb) < 2 or ta[-1] != tb[-1]:
        return False
    first_a, first_b = ta[0], tb[0]
    short, long_ = sorted((first_a, first_b), key=len)
    return len(short) >= 3 and long_.startswith(short)


def resolve_player(target: str, candidates: Iterable[str]) -> str | None:
    pool = list(candidates)
    if target in pool:
        return target
    hits = [c for c in pool if players_match(target, c)]
    return hits[0] if len(hits) == 1 else None


def is_half_line(line: Decimal) -> bool:
    return line % 1 == Decimal("0.5")


# =============================================================================
# 1. Odds conversion and devigging
# =============================================================================
def american_to_implied_prob(american_odds) -> Decimal:
    odds = to_decimal(american_odds)
    if odds == 0 or (-100 < odds < 100):
        raise ValueError(f"Invalid American odds: {american_odds}")
    if odds > 0:
        return Decimal(100) / (odds + 100)
    return -odds / (-odds + 100)


def devig_multiplicative(implied: Sequence[Decimal]) -> list[Decimal]:
    total = sum(implied, ZERO)
    if total <= 0:
        raise ValueError("Implied probabilities must sum to a positive number.")
    return [q / total for q in implied]


def devig_power(implied: Sequence[Decimal]) -> list[Decimal]:
    total = sum(implied, ZERO)
    if total <= 1:
        return devig_multiplicative(implied)

    def excess(k: Decimal) -> Decimal:
        return sum((q**k for q in implied), ZERO) - ONE

    lo, hi = ONE, Decimal(2)
    while excess(hi) > 0 and hi < Decimal(1024):
        lo, hi = hi, hi * 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _EPS:
            break
    k = (lo + hi) / 2
    powered = [q**k for q in implied]
    norm = sum(powered, ZERO)
    return [p / norm for p in powered]


def devig_shin(implied: Sequence[Decimal]) -> list[Decimal]:
    total = sum(implied, ZERO)
    if total <= 1:
        return devig_multiplicative(implied)

    def probs(z: Decimal) -> list[Decimal]:
        denom = 2 * (ONE - z)
        return [((z * z + 4 * (ONE - z) * q * q / total).sqrt() - z) / denom for q in implied]

    lo, hi = ZERO, Decimal("0.999999")
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(probs(mid), ZERO) > 1:
            lo = mid
        else:
            hi = mid
        if hi - lo < _EPS:
            break
    result = probs((lo + hi) / 2)
    norm = sum(result, ZERO)
    return [p / norm for p in result]


def devig(implied: Sequence[Decimal], method: str | None = None) -> list[Decimal]:
    method = method or config.DEVIG_METHOD
    if method == "power":
        return devig_power(implied)
    if method == "shin":
        return devig_shin(implied)
    if method == "multiplicative":
        return devig_multiplicative(implied)
    raise ValueError(f"Unknown devig method: {method}")


# =============================================================================
# 2. Multi-book weighted consensus
# =============================================================================
@dataclass(frozen=True)
class Consensus:
    probs: dict[str, Decimal]
    books_used: tuple[str, ...]
    total_weight: Decimal
    sharp_weight_share: Decimal
    dispersion: Decimal
    sharp_books_used: tuple[str, ...]


def book_weight(book_key: str) -> Decimal:
    return config.BOOK_WEIGHTS.get(book_key, config.DEFAULT_BOOK_WEIGHT)


def build_consensus(
    book_prices: Iterable[tuple[str, Mapping[str, Decimal]]],
    team_names: tuple[str, str],
    method: str | None = None,
) -> Consensus | None:
    team_a, team_b = team_names
    rows: list[tuple[str, Decimal, Decimal]] = []
    for book_key, prices in book_prices:
        if team_a not in prices or team_b not in prices:
            continue
        try:
            q_a = american_to_implied_prob(prices[team_a])
            q_b = american_to_implied_prob(prices[team_b])
            p_a, _p_b = devig([q_a, q_b], method)
        except (ValueError, ArithmeticError):
            continue
        rows.append((book_key, book_weight(book_key), p_a))
    if not rows:
        return None

    total_weight = sum((w for _, w, _ in rows), ZERO)
    p_a_mean = sum((w * p for _, w, p in rows), ZERO) / total_weight
    variance = sum((w * (p - p_a_mean) ** 2 for _, w, p in rows), ZERO) / total_weight
    sharp_rows = [(k, w) for k, w, _ in rows if k in config.SHARP_BOOKS]
    sharp_weight = sum((w for _, w in sharp_rows), ZERO)
    return Consensus(
        probs={team_a: p_a_mean, team_b: ONE - p_a_mean},
        books_used=tuple(k for k, _, _ in rows),
        total_weight=total_weight,
        sharp_weight_share=sharp_weight / total_weight,
        dispersion=variance.sqrt(),
        sharp_books_used=tuple(k for k, _ in sharp_rows),
    )


# =============================================================================
# 3. Kalshi fee model
# =============================================================================
@dataclass(frozen=True)
class FeeSchedule:
    fee_type: str = "quadratic"
    multiplier: Decimal = ONE

    def rate(self, role: str) -> Decimal:
        if role == "taker":
            return config.TAKER_FEE_BASE_RATE * self.multiplier
        if role == "maker":
            if self.fee_type == "quadratic_with_maker_fees":
                return config.MAKER_FEE_BASE_RATE * self.multiplier
            if self.fee_type == "quadratic_with_combo_maker_fees":
                return config.COMBO_MAKER_FEE_BASE_RATE * self.multiplier
            return ZERO
        raise ValueError(f"role must be 'maker' or 'taker', got {role!r}")


def fee_total(price: Decimal, contracts: int, role: str, schedule: FeeSchedule) -> Decimal:
    rate = schedule.rate(role)
    if rate == 0 or contracts <= 0:
        return ZERO
    return ceil_to_cent(rate * contracts * price * (ONE - price))


def fee_per_contract_continuous(price: Decimal, role: str, schedule: FeeSchedule) -> Decimal:
    return schedule.rate(role) * price * (ONE - price)


def fee_per_contract_exact(price: Decimal, contracts: int, role: str, schedule: FeeSchedule) -> Decimal:
    if contracts <= 0:
        return ZERO
    return fee_total(price, contracts, role, schedule) / contracts


# =============================================================================
# 4. Edge, Kelly, sizing
# =============================================================================
def net_edge(fair_prob: Decimal, price: Decimal, fee_per_contract: Decimal) -> Decimal:
    return fair_prob - (price + fee_per_contract)


def kelly_full_fraction(fair_prob: Decimal, cost_per_contract: Decimal) -> Decimal:
    if cost_per_contract <= 0 or cost_per_contract >= ONE:
        return ZERO
    fraction = (fair_prob - cost_per_contract) / (ONE - cost_per_contract)
    return fraction if fraction > 0 else ZERO


@dataclass(frozen=True)
class SizingResult:
    contracts: int
    total_cost: Decimal
    fee: Decimal
    kelly_full: Decimal
    kelly_applied: Decimal
    budget: Decimal
    binding_cap: str


def size_order(
    *,
    fair_prob: Decimal,
    price: Decimal,
    role: str,
    schedule: FeeSchedule,
    equity: Decimal,
    cash_available: Decimal,
    exposure_room: Decimal,
) -> SizingResult:
    unit_cost = price + fee_per_contract_continuous(price, role, schedule)
    k_full = kelly_full_fraction(fair_prob, unit_cost)
    k_applied = k_full * config.KELLY_FRACTION

    caps = {
        "kelly": k_applied * equity,
        "max_bet_fraction": config.MAX_BET_FRACTION * equity,
        "max_bet_dollars": config.MAX_BET_DOLLARS,
        "cash": cash_available,
        "exposure_room": exposure_room,
    }
    binding_cap = min(caps, key=lambda name: caps[name])
    budget = max(caps[binding_cap], ZERO)

    contracts = int((budget / price).to_integral_value(rounding=ROUND_FLOOR)) if price > 0 else 0
    contracts = min(contracts, config.MAX_CONTRACTS_PER_ORDER)
    while contracts > 0 and price * contracts + fee_total(price, contracts, role, schedule) > budget:
        contracts -= 1

    fee = fee_total(price, contracts, role, schedule)
    return SizingResult(
        contracts=contracts,
        total_cost=price * contracts + fee,
        fee=fee,
        kelly_full=k_full,
        kelly_applied=k_applied,
        budget=budget,
        binding_cap=binding_cap,
    )


# =============================================================================
# 5. Two-sided pricing and settlement
# =============================================================================
def price_sides(
    fair_yes: Decimal, yes_bid: Decimal, no_bid: Decimal, schedule: FeeSchedule
) -> list[dict]:
    out: list[dict] = []
    for side, bid, opposite_bid in (("yes", yes_bid, no_bid), ("no", no_bid, yes_bid)):
        fair = fair_yes if side == "yes" else ONE - fair_yes
        ask = ONE - opposite_bid
        maker_price = min(bid + config.MAKER_PRICE_IMPROVEMENT, ask - config.TICK)
        maker_price = min(max(maker_price, config.TICK), ONE - config.TICK)
        maker_edge = net_edge(fair, maker_price, fee_per_contract_continuous(maker_price, "maker", schedule))
        taker_edge = net_edge(fair, ask, fee_per_contract_continuous(ask, "taker", schedule))
        out.append({"side": side, "fair": fair, "bid": bid, "ask": ask, "maker_price": maker_price,
                    "maker_edge": maker_edge, "taker_edge": taker_edge})
    return out


def settle_payout(side: str, result: str, filled) -> Decimal:
    return Decimal(str(filled)) if result == side else ZERO


# =============================================================================
# 6. Performance & Calibration Metrics
# =============================================================================
def brier_score(predictions_vs_outcomes: Iterable[tuple[Decimal, Decimal]]) -> Decimal | None:
    """Mean squared difference between predicted probability and actual outcome (0 or 1). Lower is better."""
    records = list(predictions_vs_outcomes)
    if not records:
        return None
    return sum(((p - o) ** 2 for p, o in records), ZERO) / Decimal(len(records))


def log_loss(predictions_vs_outcomes: Iterable[tuple[Decimal, Decimal]]) -> float | None:
    """Cross-entropy loss. Lower is better."""
    records = list(predictions_vs_outcomes)
    if not records:
        return None
    loss = 0.0
    for p, o in records:
        # Clamp to avoid math.log(0)
        p_float = max(1e-15, min(1 - 1e-15, float(p)))
        o_float = float(o)
        loss += -(o_float * math.log(p_float) + (1 - o_float) * math.log(1 - p_float))
    return loss / len(records)


def closing_line_value(exec_price: Decimal, closing_price: Decimal, side: str = "yes") -> Decimal:
    """Measures edge relative to the final closing price just prior to tip-off."""
    if exec_price <= 0:
        return ZERO
    return (closing_price - exec_price) / exec_price
