"""Strict, deterministic validation of an agent's final campaign plan.

The organizer's local evaluator silently drops invalid campaigns and truncates
audiences that exceed a limit. This module checks the public environment state
before the plan is returned, without reading the hidden impact model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from scoring_core import (
    CHANNELS,
    FILTER_VALUES,
    MAX_CAMPAIGNS,
    MAX_CUSTOMERS_PER_CAMPAIGN,
)


_PROFILE_COLUMNS = {
    "ID_NUMBER",
    "arpu_segment",
    "data_segment",
    "call_segment",
    "current_tariff",
}
_CAMPAIGN_KEYS = {
    "campaign_name",
    "target_tariff",
    "channel",
    "filter_arpu_segment",
    "filter_data_segment",
    "filter_call_segment",
    "filter_current_tariff",
}


def _missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(pd.isna(value))
    return False


def _nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a nonnegative finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a nonnegative finite number")
    return number


def _current_tariffs(value: object, known_tariffs: set[str], index: int) -> list[str] | None:
    if _missing(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"Campaign {index}: filter_current_tariff must be semicolon-separated text")
    parts = [part.strip() for part in value.split(";")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Campaign {index}: filter_current_tariff has an empty tariff")
    unknown = set(parts) - known_tariffs
    if unknown:
        raise ValueError(f"Campaign {index}: unknown current tariff(s): {sorted(unknown)}")
    return parts


def _audience(profile: pd.DataFrame, campaign: dict, tariffs: list[str] | None) -> pd.DataFrame:
    """Match the documented final-campaign filter semantics of the scorer."""
    audience = profile
    for key, column in (
        ("filter_arpu_segment", "arpu_segment"),
        ("filter_data_segment", "data_segment"),
        ("filter_call_segment", "call_segment"),
    ):
        value = campaign.get(key)
        if not _missing(value):
            audience = audience[audience[column] == value]
    if tariffs is not None:
        audience = audience[audience["current_tariff"].isin(tariffs)]
    return audience.sort_values("ID_NUMBER", kind="mergesort")


def validate_plan(env, campaigns: Sequence[dict], require_pilot: bool = True) -> dict:
    """Validate final campaigns against public tariffs, audiences and remaining limits.

    Raises ``ValueError`` for any issue that would cause a silent drop, empty
    audience, or truncation in scoring. Counts include repeated contacts across
    final campaigns; the remaining budget and reach already account for pilots.
    The returned ordered IDs follow the scorer's ascending ``ID_NUMBER`` order.
    """
    if not isinstance(campaigns, list) or not 1 <= len(campaigns) <= MAX_CAMPAIGNS:
        raise ValueError(f"Expected 1 to {MAX_CAMPAIGNS} campaign dictionaries")

    history = getattr(env, "pilot_history", None)
    if not isinstance(history, list):
        raise ValueError("env.pilot_history must be a list")
    if require_pilot and not history:
        raise ValueError("At least one completed pilot is required")

    profile = getattr(env, "customer_profile", None)
    tariff_table = getattr(env, "tariffs", None)
    env_channels = getattr(env, "channels", None)
    if not isinstance(profile, pd.DataFrame) or not _PROFILE_COLUMNS.issubset(profile.columns):
        raise ValueError("env.customer_profile lacks required public columns")
    if not isinstance(tariff_table, pd.DataFrame) or "tariff_plan_code" not in tariff_table:
        raise ValueError("env.tariffs lacks tariff_plan_code")
    if not isinstance(env_channels, dict):
        raise ValueError("env.channels must be a channel dictionary")

    known_tariffs = set(tariff_table["tariff_plan_code"].dropna())
    budget_left = _nonnegative_number(getattr(env, "remaining_budget", None), "remaining_budget")
    contacts_left = _nonnegative_number(getattr(env, "remaining_contacts", None), "remaining_contacts")
    if not contacts_left.is_integer():
        raise ValueError("remaining_contacts must be an integer")
    contacts_left = int(contacts_left)

    initial_budget = budget_left
    initial_contacts = contacts_left
    total_contacts = 0
    total_cost = 0.0
    seen_ids: set = set()
    details = []

    for index, campaign in enumerate(campaigns, start=1):
        if not isinstance(campaign, dict):
            raise ValueError(f"Campaign {index}: expected a dictionary")
        unknown_keys = set(campaign) - _CAMPAIGN_KEYS
        if unknown_keys:
            raise ValueError(f"Campaign {index}: undocumented field(s): {sorted(unknown_keys)}")
        target = campaign.get("target_tariff")
        channel = campaign.get("channel")
        if not isinstance(target, str) or target not in known_tariffs:
            raise ValueError(f"Campaign {index}: unknown target_tariff {target!r}")
        if not isinstance(channel, str) or channel not in CHANNELS or channel not in env_channels:
            raise ValueError(f"Campaign {index}: unknown channel {channel!r}")

        for key, allowed in FILTER_VALUES.items():
            value = campaign.get(key)
            if not _missing(value) and (not isinstance(value, str) or value not in allowed):
                raise ValueError(f"Campaign {index}: invalid {key} {value!r}")
        tariffs = _current_tariffs(campaign.get("filter_current_tariff"), known_tariffs, index)

        audience = _audience(profile, campaign, tariffs)
        count = len(audience)
        if count == 0:
            raise ValueError(f"Campaign {index}: audience is empty")
        if count > MAX_CUSTOMERS_PER_CAMPAIGN:
            raise ValueError(
                f"Campaign {index}: {count} contacts exceed the {MAX_CUSTOMERS_PER_CAMPAIGN} per-campaign limit"
            )
        if count > contacts_left:
            raise ValueError(
                f"Campaign {index}: {count} contacts exceed {contacts_left} remaining contacts"
            )

        cost = count * CHANNELS[channel]["cost_per_contact"]
        if cost > budget_left + 1e-9:
            raise ValueError(f"Campaign {index}: cost {cost} exceeds {budget_left:g} remaining budget")

        ids = audience["ID_NUMBER"].tolist()
        new_unique = len(set(ids) - seen_ids)
        seen_ids.update(ids)
        details.append(
            {
                "name": campaign.get("campaign_name", f"campaign_{index - 1}"),
                "target_tariff": target,
                "channel": channel,
                "audience_size": count,
                "n_contacts": count,
                "cost": cost,
                "first_id": ids[0],
                "last_id": ids[-1],
                "selected_ids": ids,
                "n_new_unique_customers": new_unique,
                "n_repeat_contacts": count - new_unique,
            }
        )
        contacts_left -= count
        budget_left -= cost
        total_contacts += count
        total_cost += cost

    return {
        "n_campaigns": len(campaigns),
        "n_pilots": len(history),
        "total_contacts": total_contacts,
        "total_cost": total_cost,
        "n_unique_customers": len(seen_ids),
        "n_repeat_contacts": total_contacts - len(seen_ids),
        "remaining_contacts": contacts_left,
        "remaining_budget": budget_left,
        "initial_remaining_contacts": initial_contacts,
        "initial_remaining_budget": initial_budget,
        "campaigns_detail": details,
    }
