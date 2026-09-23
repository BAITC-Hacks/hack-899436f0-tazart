"""Select segment/tariff hypotheses using only information available to the agent.

``prior_mean`` is a *channel-neutral* expected relative ARPU lift per offer:
an estimated ARPU change conditional on migration times a heuristic offer-response
probability. Thus a push campaign starts at 0.50 * prior_mean and an SMS campaign
at 0.65 * prior_mean. The historical file contains migrations, not offers or
rejections; its target frequencies must never be interpreted as conversion rates.
The deliberately wide ``prior_sd`` reflects this mismatch and lets pilots
override historical suggestions.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


_HISTORY_PATH = Path(__file__).resolve().parent / "data" / "change_tariff.csv"
_SEGMENTS = {"LOW", "MID", "HIGH"}


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _history() -> dict[tuple[str, str, str], list[float]]:
    """Deduplicate migrations and suppress unstable relative changes."""
    changes: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    if not _HISTORY_PATH.is_file():
        return changes

    seen: set[tuple[str, str, str, str]] = set()
    with _HISTORY_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source = row.get("tariff_plan_code_from", "")
            target = row.get("tariff_plan_code_to", "")
            key = (row.get("ID_NUMBER", ""), row.get("TIME_KEY", ""), source, target)
            if key in seen:
                continue
            seen.add(key)
            before = _number(row.get("AVG_ARPU_PREV_3M"), float("nan"))
            after = _number(row.get("AVG_ARPU_NEXT_3M"), float("nan"))
            if before < 100 or not math.isfinite(before) or not math.isfinite(after):
                continue
            segment = "LOW" if before <= 1000 else "MID" if before <= 5000 else "HIGH"
            change = _clip((after - before) / before, -1.0, 3.0)
            changes[(source, segment, target)].append(change)
    return changes


def _tariff_features(env) -> dict[str, tuple[float, float, float]]:
    features: dict[str, tuple[float, float, float]] = {}
    for _, row in env.tariffs.iterrows():
        code = str(row["tariff_plan_code"])
        price = max(0.0, _number(row.get("price_tariff")))
        data = max(0.0, _number(row.get("Data_in_PKG")))
        minutes = max(0.0, _number(row.get("Min_another_operator_in_PKG")))
        minutes += max(0.0, _number(row.get("Min_another_operator_and_city_in_PKG")))
        features[code] = (price, data, minutes)
    return features


def _conditional_change(
    source: tuple[float, float, float],
    target: tuple[float, float, float],
    median_price: float,
) -> float:
    """Weak price/package estimate for transitions absent from history."""
    old_price, old_data, old_minutes = source
    new_price, new_data, new_minutes = target
    price_change = (new_price - old_price) / median_price
    data_change = math.log1p(new_data / 2048) - math.log1p(old_data / 2048)
    minute_change = math.log1p(new_minutes / 50) - math.log1p(old_minutes / 50)
    estimate = 0.38 * price_change + 0.045 * data_change + 0.025 * minute_change
    return _clip(estimate, -0.65, 1.0)


def _offer_response_proxy(source_price: float, target_price: float, median_price: float) -> float:
    """Planning assumption, intentionally unrelated to historical target share."""
    premium = max(0.0, target_price - source_price) / median_price
    discount = max(0.0, source_price - target_price) / median_price
    return _clip(0.22 - 0.045 * premium + 0.015 * discount, 0.10, 0.25)


def build_candidates(env, limit: int = 80) -> list[dict]:
    """Return deterministic, diverse hypotheses for (current tariff, ARPU, target).

    Every returned filter matches one disjoint base cell with at most 5,000
    customers. Targets are drawn from the tariff catalog, including transitions
    with no history. The output is a shortlist for pilots and portfolio planning,
    not a claim that the historical migration rate predicts offer acceptance.
    """
    if limit <= 0:
        return []

    features = _tariff_features(env)
    if not features:
        return []
    positive_prices = [value[0] for value in features.values() if value[0] > 0]
    median_price = max(1.0, median(positive_prices)) if positive_prices else 1.0
    history = _history()

    profile = env.customer_profile
    cells = (
        profile.groupby(["current_tariff", "arpu_segment"], observed=True)
        .agg(segment_size=("ID_NUMBER", "size"), segment_arpu_sum=("predicted_arpu", "sum"))
        .reset_index()
    )

    by_cell: list[tuple[float, str, str, list[dict]]] = []
    for _, cell in cells.iterrows():
        source_code = str(cell["current_tariff"])
        segment = str(cell["arpu_segment"])
        if source_code not in features or segment not in _SEGMENTS:
            continue
        size = int(cell["segment_size"])
        if size <= 0 or size > 5000:
            continue
        arpu_sum = max(0.0, _number(cell["segment_arpu_sum"]))
        if arpu_sum <= 0:
            continue

        source = features[source_code]
        options: list[dict] = []
        for target_code, target in features.items():
            if target_code == source_code:
                continue
            fallback = _conditional_change(source, target, median_price)
            samples = history.get((source_code, segment, target_code), [])
            count = len(samples)
            if count:
                # A median is less sensitive to the surviving capped outliers.
                # Saturating evidence weight limits domain shift from historical
                # migrants to the current offer audience.
                weight = min(0.72, count / (count + 35.0))
                conditional_change = weight * median(samples) + (1 - weight) * fallback
                origin = "history_shrunk"
            else:
                conditional_change = fallback
                origin = "price_package_heuristic"

            response = _offer_response_proxy(source[0], target[0], median_price)
            theta = _clip(conditional_change * response, -0.3, 0.75)
            # Even hundreds of observed migrations do not identify an offer
            # conversion probability. A wide floor prevents false certainty.
            sd = max(0.22, 0.30 - 0.025 * min(math.log1p(count), 3.0))
            options.append({
                "filter_current_tariff": source_code,
                "filter_arpu_segment": segment,
                "target_tariff": target_code,
                "segment_size": size,
                "segment_arpu_sum": arpu_sum,
                "prior_mean": float(theta),
                "prior_sd": float(sd),
                "history_count": count,
                "prior_source": origin,
            })

        # Use predicted portfolio value, with a modest discovery premium.
        options.sort(
            key=lambda item: (
                -(item["prior_mean"] + 0.15 * item["prior_sd"]) * arpu_sum,
                -item["history_count"],
                item["target_tariff"],
            )
        )
        if options:
            cell_score = max(0.0, options[0]["prior_mean"] + 0.15 * options[0]["prior_sd"]) * arpu_sum
            by_cell.append((cell_score, source_code, segment, options))

    by_cell.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[dict] = []
    # Round-robin gives high-value cells alternatives without filling the list
    # with many targets for one enormous cell.
    for round_index in range(max((len(item[3]) for item in by_cell), default=0)):
        for _, _, _, options in by_cell:
            if round_index < len(options):
                selected.append(options[round_index])
                if len(selected) >= limit:
                    return selected
    return selected
