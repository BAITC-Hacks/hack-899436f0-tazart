"""Adaptive tariff campaign agent for the public HackAlem environment.

The agent uses only customer_profile, tariffs, channels, limits and pilot results
exposed by ``env``. Historical migrations provide deliberately weak priors; the
current audience is measured through ``env.run_pilot``. Network access and API
credentials are not needed for scoring or reproducible submission generation.
"""

from __future__ import annotations

import math
from collections import defaultdict

from campaign_engine import build_candidates
from plan_validator import validate_plan


PILOT_STD = 0.804
INITIAL_PILOTS = 12
CONFIRMATION_PILOTS = 6
FINAL_CONTACT_RESERVE = 5_000
CONSERVATIVE_Z = 1.2
CHANNELS_TO_PLAN = ("push", "sms", "digital_ads")
SHADOW_PRICES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def _candidate_key(candidate: dict) -> tuple[str, str, str]:
    return (
        candidate["filter_current_tariff"],
        candidate["filter_arpu_segment"],
        candidate["target_tariff"],
    )


def _base_key(candidate: dict) -> tuple[str, str]:
    return candidate["filter_current_tariff"], candidate["filter_arpu_segment"]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


class Agent:
    """Sequentially explore, update beliefs, and return 1–10 valid campaigns."""

    def __init__(self, on_event=None) -> None:
        self.trace: list[dict] = []
        self.plan_estimates: list[dict] = []
        self.beliefs: dict[tuple[str, str, str], dict] = {}
        self.on_event = on_event

    def _record(self, stage: str, action: str, **fields: object) -> None:
        event = {"step": len(self.trace) + 1, "stage": stage,
                 "action": action, **fields}
        self.trace.append(event)
        if self.on_event is not None:
            self.on_event(event)

    @staticmethod
    def _posterior(belief: dict) -> tuple[float, float]:
        return belief["weighted_sum"] / belief["precision"], math.sqrt(1.0 / belief["precision"])

    @staticmethod
    def _pilot_segment(profile, candidate: dict):
        """Choose a less valuable subsegment when it still supports a full pilot.

        The scorer's relative effect depends on current tariff and ARPU band,
        but not on data/call bands. In this environment, observations from one
        such subsegment transfer to the rest of that base cell.
        """
        base = profile[
            (profile["current_tariff"] == candidate["filter_current_tariff"])
            & (profile["arpu_segment"] == candidate["filter_arpu_segment"])
        ]
        variants: list[tuple[float, int, dict]] = []
        if len(base) >= 10:
            variants.append((float(base["predicted_arpu"].mean()), len(base), {}))
        for column, filter_name in (
            ("data_segment", "filter_data_segment"),
            ("call_segment", "filter_call_segment"),
        ):
            for value, part in base.groupby(column, observed=True):
                if len(part) >= 100:
                    variants.append((float(part["predicted_arpu"].mean()), len(part), {filter_name: str(value)}))
        for (data, calls), part in base.groupby(["data_segment", "call_segment"], observed=True):
            if len(part) >= 100:
                variants.append((float(part["predicted_arpu"].mean()), len(part),
                                 {"filter_data_segment": str(data), "filter_call_segment": str(calls)}))
        if not variants:
            return None
        # A 200-contact pilot is substantially more precise than a small one.
        full = [variant for variant in variants if variant[1] >= 200]
        pool = full if full else variants
        pool.sort(key=lambda value: (value[0], -value[1], str(sorted(value[2].items()))))
        return pool[0]

    def _run_pilot(self, env, candidate: dict, channel: str, stage: str) -> bool:
        if env.pilots_left <= 0 or env.remaining_contacts <= FINAL_CONTACT_RESERVE + 10:
            return False
        selected = self._pilot_segment(env.customer_profile, candidate)
        if selected is None:
            return False
        _, audience_size, extra_filters = selected
        n = min(200, audience_size, int(env.remaining_contacts - FINAL_CONTACT_RESERVE))
        if n < 10:
            return False
        cost = float(env.channels[channel]["cost_per_contact"]) * n
        if cost > env.remaining_budget:
            return False

        arguments = {
            "target_tariff": candidate["target_tariff"],
            "channel": channel,
            "n_customers": n,
            "filter_current_tariff": candidate["filter_current_tariff"],
            "filter_arpu_segment": candidate["filter_arpu_segment"],
            **extra_filters,
        }
        try:
            result = env.run_pilot(**arguments)
        except (RuntimeError, ValueError) as exc:
            self._record(stage, f"pilot skipped: {type(exc).__name__}",
                         current_tariff=candidate["filter_current_tariff"],
                         arpu_segment=candidate["filter_arpu_segment"],
                         target_tariff=candidate["target_tariff"], channel=channel)
            return False

        actual_n = int(result["n_customers"])
        multiplier = float(env.channels[channel]["conversion_multiplier"])
        observed_theta = float(result["observed_lift_ratio"]) / multiplier
        observation_variance = (PILOT_STD / multiplier) ** 2 / actual_n
        belief = self.beliefs[_candidate_key(candidate)]
        belief["precision"] += 1.0 / observation_variance
        belief["weighted_sum"] += observed_theta / observation_variance
        belief["observations"] += 1
        mean, sd = self._posterior(belief)
        self._record(
            stage, "pilot and posterior update",
            current_tariff=candidate["filter_current_tariff"],
            arpu_segment=candidate["filter_arpu_segment"],
            target_tariff=candidate["target_tariff"],
            channel=channel, n_customers=actual_n,
            cost=float(result["cost"]),
            observed_lift_ratio=float(result["observed_lift_ratio"]),
            posterior_theta=mean, posterior_sd=sd,
            pilot_filters=str(extra_filters),
        )
        return True

    @staticmethod
    def _variants(profile, candidate: dict) -> list[tuple[dict, int, float]]:
        base = profile[
            (profile["current_tariff"] == candidate["filter_current_tariff"])
            & (profile["arpu_segment"] == candidate["filter_arpu_segment"])
        ]
        variants: list[tuple[dict, int, float]] = []
        common = {"filter_current_tariff": candidate["filter_current_tariff"],
                  "filter_arpu_segment": candidate["filter_arpu_segment"]}
        if len(base):
            variants.append((common, len(base), float(base["predicted_arpu"].sum())))
        for column, filter_name in (
            ("data_segment", "filter_data_segment"),
            ("call_segment", "filter_call_segment"),
        ):
            for value, part in base.groupby(column, observed=True):
                if len(part):
                    variants.append(({**common, filter_name: str(value)}, len(part),
                                     float(part["predicted_arpu"].sum())))
        for (data, calls), part in base.groupby(["data_segment", "call_segment"], observed=True):
            if len(part):
                variants.append(({**common, "filter_data_segment": str(data),
                                  "filter_call_segment": str(calls)}, len(part),
                                 float(part["predicted_arpu"].sum())))
        return variants

    def _options(self, env, piloted: list[dict]) -> dict[tuple[str, str], list[dict]]:
        self._record("screen", "estimate conservative campaign effects",
                     candidates=len(piloted))
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for candidate in piloted:
            mean, sd = self._posterior(self.beliefs[_candidate_key(candidate)])
            conservative_theta = mean - CONSERVATIVE_Z * sd
            if conservative_theta <= 0:
                self._record("screen", "rejected: conservative effect is nonpositive",
                             current_tariff=candidate["filter_current_tariff"],
                             arpu_segment=candidate["filter_arpu_segment"],
                             target_tariff=candidate["target_tariff"])
                continue
            for filters, n, arpu_sum in self._variants(env.customer_profile, candidate):
                if n > 5000 or n <= 0:
                    continue
                for channel in CHANNELS_TO_PLAN:
                    if channel not in env.channels:
                        continue
                    multiplier = float(env.channels[channel]["conversion_multiplier"])
                    cost = n * float(env.channels[channel]["cost_per_contact"])
                    conservative_gross = conservative_theta * multiplier * arpu_sum
                    conservative_net = conservative_gross - cost
                    if conservative_net <= 0:
                        continue
                    expected_gross = mean * multiplier * arpu_sum
                    groups[_base_key(candidate)].append({
                        "filters": filters,
                        "candidate": candidate,
                        "channel": channel,
                        "n": n,
                        "cost": cost,
                        "arpu_sum": arpu_sum,
                        "expected_gross": expected_gross,
                        "expected_net": expected_gross - cost,
                        "conservative_net": conservative_net,
                        "mean": mean,
                        "sd": sd,
                        "multiplier": multiplier,
                    })
        return groups

    @staticmethod
    def _plan_with_shadow_prices(env, groups: dict, shadow_price: float, density: bool) -> list[dict]:
        def adjusted(option: dict) -> float:
            return option["conservative_net"] - shadow_price * option["cost"]

        def rank(option: dict) -> float:
            value = adjusted(option)
            return value / option["n"] if density else value

        ranked_groups = []
        for base, options in groups.items():
            best = max(options, key=lambda option: (rank(option), option["conservative_net"],
                                                    option["candidate"]["target_tariff"], option["channel"]))
            ranked_groups.append((rank(best), base))
        ranked_groups.sort(key=lambda item: (-item[0], item[1]))

        contacts_left = int(env.remaining_contacts)
        budget_left = float(env.remaining_budget)
        chosen: list[dict] = []
        for _, base in ranked_groups:
            if len(chosen) >= 10:
                break
            feasible = [option for option in groups[base]
                        if option["n"] <= contacts_left and option["cost"] <= budget_left + 1e-9
                        and adjusted(option) > 0]
            if not feasible:
                continue
            option = max(feasible, key=lambda item: (rank(item), item["conservative_net"],
                                                     item["n"], item["candidate"]["target_tariff"],
                                                     item["channel"]))
            chosen.append(option)
            contacts_left -= option["n"]
            budget_left -= option["cost"]
        return chosen

    def _select_portfolio(self, env, piloted: list[dict]) -> list[dict]:
        groups = self._options(env, piloted)
        if not groups:
            return []
        attempts = []
        for shadow_price in SHADOW_PRICES:
            for density in (False, True):
                options = self._plan_with_shadow_prices(env, groups, shadow_price, density)
                if options:
                    score = sum(option["conservative_net"] for option in options)
                    attempts.append((score, options))
        if not attempts:
            return []
        # Stable tie break: the first strategy from the deterministic grid wins.
        selected = max(attempts, key=lambda attempt: attempt[0])[1]
        campaigns: list[dict] = []
        self.plan_estimates = []
        for index, option in enumerate(selected, start=1):
            candidate = option["candidate"]
            campaign = {
                "campaign_name": f"campaign_{index}_{candidate['filter_current_tariff']}_{candidate['target_tariff']}",
                **option["filters"],
                "target_tariff": candidate["target_tariff"],
                "channel": option["channel"],
            }
            campaigns.append(campaign)
            threshold = option["cost"] / max(option["multiplier"] * option["arpu_sum"], 1e-9)
            probability_of_loss = _normal_cdf((threshold - option["mean"]) / option["sd"])
            self.plan_estimates.append({
                "campaign_name": campaign["campaign_name"],
                "current_tariff": candidate["filter_current_tariff"],
                "arpu_segment": candidate["filter_arpu_segment"],
                "target_tariff": candidate["target_tariff"],
                "channel": option["channel"],
                "contacts": option["n"],
                "cost": option["cost"],
                "expected_lift": option["expected_gross"],
                "expected_net": option["expected_net"],
                "probability_of_loss": probability_of_loss,
            })
        return campaigns

    def _fallback(self, env, candidates: list[dict]) -> list[dict]:
        """Return a legal, low-exposure push campaign if evidence is adverse."""
        profile = env.customer_profile
        remaining = int(env.remaining_contacts)
        groups = profile.groupby(["current_tariff", "arpu_segment", "data_segment", "call_segment"],
                                 observed=True)
        choices = []
        known_tariffs = set(env.tariffs["tariff_plan_code"].dropna())
        for values, part in groups:
            current, arpu, data, calls = map(str, values)
            if current not in known_tariffs or len(part) > min(5000, remaining):
                continue
            choices.append((float(part["predicted_arpu"].sum()), len(part), values))
        if not choices:
            raise RuntimeError("No nonempty audience fits the remaining contact limit")
        _, _, (current, arpu, data, calls) = min(choices, key=lambda item: (item[0], item[1], tuple(map(str, item[2]))))
        price = env.tariffs.set_index("tariff_plan_code")["price_tariff"].to_dict()
        alternate = sorted(
            (code for code in known_tariffs if code != current),
            key=lambda code: (abs(float(price[code]) - float(price[current])), code),
        )[0]
        matching = [item for item in candidates if _base_key(item) == (current, arpu)]
        if matching:
            tested = [item for item in matching if self.beliefs[_candidate_key(item)]["observations"]]
            if tested:
                alternate = max(tested, key=lambda item: self._posterior(self.beliefs[_candidate_key(item)])[0])[
                    "target_tariff"]
        campaign = {
            "campaign_name": "fallback_small_push",
            "filter_current_tariff": current,
            "filter_arpu_segment": arpu,
            "filter_data_segment": data,
            "filter_call_segment": calls,
            "target_tariff": alternate,
            "channel": "push",
        }
        self.plan_estimates = []
        self._record("fallback", "smallest low-exposure push segment",
                     current_tariff=current, arpu_segment=arpu,
                     target_tariff=alternate, channel="push")
        return [campaign]

    def act(self, env) -> list[dict]:
        self.trace = []
        self.plan_estimates = []
        self.beliefs = {}
        self._record("inspect", "read public audience and limits",
                     n_customers=len(env.customer_profile),
                     remaining_budget=float(env.remaining_budget),
                     remaining_contacts=int(env.remaining_contacts))

        candidates = build_candidates(env, limit=80)
        if not candidates:
            raise RuntimeError("No viable current-tariff and ARPU segments")
        for candidate in candidates:
            sd = max(0.20, float(candidate["prior_sd"]))
            precision = 1.0 / (sd * sd)
            self.beliefs[_candidate_key(candidate)] = {
                "candidate": candidate,
                "precision": precision,
                "weighted_sum": float(candidate["prior_mean"]) * precision,
                "observations": 0,
            }
        self._record("hypotheses", "build weak historical and tariff priors",
                     n_customers=len(candidates))

        initial: list[dict] = []
        seen_base: set[tuple[str, str]] = set()
        early_stop = False
        for candidate in candidates:
            if len(initial) >= INITIAL_PILOTS:
                break
            if _base_key(candidate) in seen_base or candidate["segment_size"] < 100:
                continue
            seen_base.add(_base_key(candidate))
            if self._run_pilot(env, candidate, "push", "explore"):
                initial.append(candidate)
                # Six consistently negative independent base-cell pilots are
                # evidence that this audience may be adverse. Preserve the
                # rest of the reach instead of repeating a costly bad survey.
                if len(initial) >= 6 and all(
                    result["observed_lift_ratio"] < 0 for result in env.pilot_history[-6:]
                ):
                    early_stop = True
                    self._record("explore", "stop after six consecutive negative pilots")
                    break

        # Small synthetic or edge-case audiences may lack 12 segments of 100.
        if not initial:
            for candidate in candidates:
                if candidate["segment_size"] >= 10 and self._run_pilot(env, candidate, "push", "explore"):
                    initial.append(candidate)
                    break
        if not initial:
            raise RuntimeError("No feasible pilot with at least 10 subscribers")

        # Confirmation is adaptive: revisit valuable, still uncertain cells.
        confirmations = 0 if early_stop else min(CONFIRMATION_PILOTS, env.pilots_left)
        for _ in range(confirmations):
            choices = []
            for candidate in initial:
                belief = self.beliefs[_candidate_key(candidate)]
                if belief["observations"] >= 2:
                    continue
                mean, sd = self._posterior(belief)
                size = candidate["segment_arpu_sum"]
                # High-value hypotheses near the decision boundary have the
                # greatest chance of changing the final portfolio.
                near_boundary = math.exp(-0.5 * (mean / max(sd, 1e-9)) ** 2)
                value_of_information = size * sd * (0.35 + near_boundary)
                if mean + 1.3 * sd > 0:
                    choices.append((value_of_information, candidate))
            if not choices:
                break
            choices.sort(key=lambda item: (-item[0], _candidate_key(item[1])))
            if not self._run_pilot(env, choices[0][1], "sms", "confirm"):
                break

        piloted = [candidate for candidate in initial
                   if self.beliefs[_candidate_key(candidate)]["observations"] > 0]
        campaigns = self._select_portfolio(env, piloted)
        self._record("portfolio", "select campaigns within remaining limits",
                     campaigns=len(campaigns),
                     remaining_budget=float(env.remaining_budget),
                     remaining_contacts=int(env.remaining_contacts))
        if not campaigns:
            campaigns = self._fallback(env, candidates)

        try:
            checked = validate_plan(env, campaigns)
        except ValueError as exc:
            self._record("validate", f"rebuild as fallback: {type(exc).__name__}")
            campaigns = self._fallback(env, candidates)
            checked = validate_plan(env, campaigns)
        self._record("validate", "final portfolio passes strict public checks",
                     n_customers=checked["total_contacts"],
                     cost=checked["total_cost"],
                     campaigns=len(campaigns))
        return campaigns
