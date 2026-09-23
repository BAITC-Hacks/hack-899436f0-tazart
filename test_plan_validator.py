"""Focused tests for strict campaign validation, independent of impact values."""

import unittest
from types import SimpleNamespace

import pandas as pd

from plan_validator import validate_plan
from scoring_core import CHANNELS


def make_env():
    profile = pd.DataFrame(
        [
            (3, "HIGH", "LITE", "LOW", "tariff_1"),
            (1, "HIGH", "HEAVY", "LOW", "tariff_1"),
            (4, "MID", "LITE", "HIGH", "tariff_2"),
            (2, "MID", "HEAVY", "HIGH", "tariff_1"),
        ],
        columns=["ID_NUMBER", "arpu_segment", "data_segment", "call_segment", "current_tariff"],
    )
    return SimpleNamespace(
        customer_profile=profile,
        tariffs=pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_2"]}),
        channels=CHANNELS,
        remaining_budget=10,
        remaining_contacts=5,
        pilot_history=[{"n_customers": 10, "cost": 40}],
    )


class ValidatePlanTests(unittest.TestCase):
    def test_counts_repeated_contacts_in_scorer_id_order(self):
        env = make_env()
        plan = [
            {"target_tariff": "tariff_2", "channel": "sms", "filter_arpu_segment": "HIGH"},
            {"target_tariff": "tariff_2", "channel": "push", "filter_current_tariff": "tariff_1"},
        ]
        report = validate_plan(env, plan)
        self.assertEqual(report["total_contacts"], 5)
        self.assertEqual(report["total_cost"], 8)
        self.assertEqual(report["n_unique_customers"], 3)
        self.assertEqual(report["campaigns_detail"][0]["selected_ids"], [1, 3])
        self.assertEqual(report["campaigns_detail"][1]["selected_ids"], [1, 2, 3])
        self.assertEqual(report["remaining_contacts"], 0)

    def test_rejects_silent_truncation_and_bad_filter(self):
        env = make_env()
        env.remaining_contacts = 1
        with self.assertRaisesRegex(ValueError, "remaining contacts"):
            validate_plan(env, [{"target_tariff": "tariff_2", "channel": "push", "filter_arpu_segment": "HIGH"}])
        with self.assertRaisesRegex(ValueError, "empty tariff"):
            validate_plan(make_env(), [{"target_tariff": "tariff_2", "channel": "push", "filter_current_tariff": "tariff_1;;tariff_2"}])
        with self.assertRaisesRegex(ValueError, "invalid filter_data_segment"):
            validate_plan(make_env(), [{"target_tariff": "tariff_2", "channel": "push", "filter_data_segment": "INVALID"}])

    def test_rejects_missing_pilot_and_budget_overrun(self):
        env = make_env()
        env.pilot_history = []
        with self.assertRaisesRegex(ValueError, "pilot"):
            validate_plan(env, [{"target_tariff": "tariff_2", "channel": "push", "filter_arpu_segment": "HIGH"}])
        env.pilot_history = [{"n_customers": 10}]
        with self.assertRaisesRegex(ValueError, "remaining budget"):
            validate_plan(env, [{"target_tariff": "tariff_2", "channel": "call", "filter_arpu_segment": "HIGH"}])

    def test_rejects_internal_and_undocumented_fields(self):
        for field, value in (("explicit_ids", [1, 4]), ("filter_tariff", "tariff_1")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "undocumented field"):
                validate_plan(
                    make_env(),
                    [{"target_tariff": "tariff_2", "channel": "push", field: value}],
                )


if __name__ == "__main__":
    unittest.main()
