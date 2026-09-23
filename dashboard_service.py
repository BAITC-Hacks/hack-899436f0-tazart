"""Data adapter for the local Beeline campaign command centre.

The agent receives only the public environment. The mock impact model is used
by the environment's private pilot closure and by the official scorer *after*
the agent has returned its plan. Values derived from the mock scorer are kept
separate from the agent's estimates in the returned payload.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Callable

import pandas as pd

from agent import Agent
from environment import make_environment
from mock_environment import (
    CHANNELS,
    MAX_TOTAL_CONTACTS,
    TOTAL_BUDGET,
    _mock_fallback,
    _mock_impact_model,
)
from plan_validator import validate_plan
from scoring_core import score_campaigns


ROOT = Path(__file__).resolve().parent
FILTER_KEYS = (
    "filter_current_tariff",
    "filter_arpu_segment",
    "filter_data_segment",
    "filter_call_segment",
)


@lru_cache(maxsize=1)
def _public_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load organizer-provided public data once, without modifying the files."""
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    history = pd.read_csv(ROOT / "data" / "change_tariff.csv")
    return profile, tariffs, history


def _audience_map(profile: pd.DataFrame) -> list[dict]:
    """Actual current-audience counts by disjoint tariff × ARPU cells."""
    grouped = (
        profile.groupby(["current_tariff", "arpu_segment"], observed=True)
        .agg(audience_count=("ID_NUMBER", "size"), baseline_arpu=("predicted_arpu", "sum"))
        .reset_index()
        .sort_values(["current_tariff", "arpu_segment"], kind="mergesort")
    )
    return [
        {
            "current_tariff": str(row.current_tariff),
            "arpu_segment": str(row.arpu_segment),
            "audience_count": int(row.audience_count),
            "baseline_arpu": float(row.baseline_arpu),
        }
        for row in grouped.itertuples(index=False)
    ]


def get_overview() -> dict:
    """Return pre-run facts displayed before the user starts a scenario."""
    profile, tariffs, _ = _public_tables()
    return {
        "audience_count": int(len(profile)),
        "baseline_arpu": float(profile["predicted_arpu"].sum()),
        "max_budget": int(TOTAL_BUDGET),
        "max_contacts": int(MAX_TOTAL_CONTACTS),
        "tariff_count": int(tariffs["tariff_plan_code"].nunique()),
        "segment_map": _audience_map(profile),
    }


def _validated_budget(budget: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise ValueError("Budget must be an integer from 0 to 100000")
    budget = int(budget)
    if not 0 <= budget <= TOTAL_BUDGET:
        raise ValueError("Budget must be an integer from 0 to 100000")
    return budget


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("Seed must be a nonnegative integer")
    return int(seed)


def _campaign_rows(campaigns: list[dict], checked: dict, estimates: list[dict]) -> list[dict]:
    estimate_by_name = {item["campaign_name"]: item for item in estimates}
    result: list[dict] = []
    for campaign, detail in zip(campaigns, checked["campaigns_detail"], strict=True):
        estimate = estimate_by_name.get(campaign["campaign_name"])
        filters = {key: campaign[key] for key in FILTER_KEYS if campaign.get(key) is not None}
        result.append({
            "campaign_name": str(campaign["campaign_name"]),
            "filters": filters,
            "current_tariff": campaign.get("filter_current_tariff"),
            "arpu_segment": campaign.get("filter_arpu_segment"),
            "data_segment": campaign.get("filter_data_segment"),
            "call_segment": campaign.get("filter_call_segment"),
            "target_tariff": str(campaign["target_tariff"]),
            "channel": str(campaign["channel"]),
            "matched": int(detail["audience_size"]),
            "contacts": int(detail["n_contacts"]),
            "cost": float(detail["cost"]),
            "new_unique_customers": int(detail["n_new_unique_customers"]),
            "repeat_contacts": int(detail["n_repeat_contacts"]),
            "expected_lift": float(estimate["expected_lift"]) if estimate else None,
            "expected_net": float(estimate["expected_net"]) if estimate else None,
            "probability_of_loss": float(estimate["probability_of_loss"]) if estimate else None,
        })
    return result


def _pilot_rows(history: list[dict], events: list[dict]) -> list[dict]:
    observations = [event for event in events if event.get("action") == "pilot and posterior update"]
    rows: list[dict] = []
    for index, pilot in enumerate(history):
        item = dict(pilot)
        if index < len(observations):
            event = observations[index]
            item.update({
                "stage": event.get("stage"),
                "current_tariff": event.get("current_tariff"),
                "arpu_segment": event.get("arpu_segment"),
                "posterior_theta": event.get("posterior_theta"),
                "posterior_sd": event.get("posterior_sd"),
            })
        rows.append(item)
    return rows


def _score_mock(
    pilot_campaigns: list[dict],
    final_campaigns: list[dict],
    profile: pd.DataFrame,
    tariffs: pd.DataFrame,
    history: pd.DataFrame,
) -> dict:
    """Call the organizer's public scorer only after the agent has finished."""
    strategy = pd.DataFrame(pilot_campaigns + final_campaigns)
    for column in (*FILTER_KEYS, "explicit_ids"):
        if column not in strategy:
            strategy[column] = None
    mock_model = _mock_impact_model(history)
    return score_campaigns(
        strategy,
        profile,
        mock_model,
        tariffs,
        float(profile["predicted_arpu"].sum()),
        _mock_fallback,
        team_id="local-dashboard",
    )


def run_scenario(
    budget: int,
    seed: int = 42,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Run the real agent against a fresh local mock environment.

    ``budget`` is applied to the environment before any pilot; it is not a UI
    decoration. The scorer's global cap is 100000, but every campaign and pilot
    is already checked against the smaller selected budget when applicable.
    """
    budget = _validated_budget(budget)
    seed = _validated_seed(seed)
    profile, tariffs, history = _public_tables()

    # This impact table enters only the private environment closure. The Agent
    # receives solely its public `env` object, never this model or the scorer.
    env, internals = make_environment(
        customer_profile=profile,
        impact_model=_mock_impact_model(history),
        dict_tariff=tariffs,
        channels=CHANNELS,
        total_budget=budget,
        max_total_contacts=MAX_TOTAL_CONTACTS,
        fallback_predict=_mock_fallback,
        seed=seed,
    )
    agent = Agent(on_event=on_event)
    campaigns = agent.act(env)
    checked = validate_plan(env, campaigns)

    pilots = _pilot_rows(env.pilot_history, agent.trace)
    final_rows = _campaign_rows(campaigns, checked, agent.plan_estimates)
    pilot_contacts = sum(int(item["n_customers"]) for item in env.pilot_history)
    pilot_cost = sum(float(item["cost"]) for item in env.pilot_history)
    final_contacts = int(checked["total_contacts"])
    final_cost = float(checked["total_cost"])
    budget_used = pilot_cost + final_cost
    contacts_used = pilot_contacts + final_contacts
    if budget_used > budget + 1e-9 or contacts_used > MAX_TOTAL_CONTACTS:
        raise RuntimeError("Scenario exceeded the public budget or contact cap")

    rejected = [
        {**event, "reason": event["action"].removeprefix("rejected: ")}
        for event in agent.trace
        if event.get("stage") == "screen" and str(event.get("action", "")).startswith("rejected: ")
    ]
    mock_result = _score_mock(
        internals.executed_pilot_campaigns(), campaigns, profile, tariffs, history
    )
    if int(mock_result["total_contacts"]) != contacts_used or abs(float(mock_result["total_cost"]) - budget_used) > 1e-9:
        raise RuntimeError("Official scorer and strict validator disagree on contacts or cost")

    totals = {
        "budget": budget,
        "budget_used": budget_used,
        "budget_remaining": float(env.remaining_budget) - final_cost,
        "contacts_limit": int(MAX_TOTAL_CONTACTS),
        "contacts_used": contacts_used,
        "contacts_remaining": int(env.remaining_contacts) - final_contacts,
        "pilot_contacts": pilot_contacts,
        "pilot_cost": pilot_cost,
        "final_contacts": final_contacts,
        "final_cost": final_cost,
        "campaign_count": len(campaigns),
        "pilot_count": len(pilots),
        "audience_count": int(len(profile)),
        "baseline_arpu": float(profile["predicted_arpu"].sum()),
    }
    return {
        "seed": seed,
        "totals": totals,
        "campaigns": final_rows,
        "pilots": pilots,
        "events": [dict(event) for event in agent.trace],
        "rejected": rejected,
        "agent_estimated_final_net": (
            sum(float(item["expected_net"]) for item in agent.plan_estimates)
            if agent.plan_estimates else None
        ),
        "mock_result": mock_result,
        "segment_map": _audience_map(profile),
    }
