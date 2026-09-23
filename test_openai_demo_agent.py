"""Protocol and privacy checks for the optional Responses API path."""

from __future__ import annotations

import io
import html
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from openai_demo_agent import run_openai_demo


def _local_result() -> dict:
    return {
        "totals": {
            "pilot_count": 2,
            "pilot_contacts": 400,
            "pilot_cost": 0,
            "campaign_count": 1,
            "final_contacts": 100,
            "final_cost": 300,
            "budget_used": 300,
            "budget_remaining": 700,
            "contacts_remaining": 14_500,
        },
        "pilots": [
            {"observed_lift_ratio": 0.1, "ID_NUMBER": "must-stay-local"},
            {"observed_lift_ratio": -0.1, "ID_NUMBER": "must-stay-local"},
        ],
        "campaigns": [
            {"channel": "sms", "filters": {"explicit_ids": ["must-stay-local"]}},
        ],
        "rejected": [{"reason": "must-stay-local"}],
        "agent_estimated_final_net": 200,
        "mock_result": {"secret": "must-stay-local"},
    }


def _api_response(body: dict, index: int) -> dict:
    names = ("check_hypotheses", "analyze_segments", "calculate_campaigns")
    if index < 3:
        assert body["tool_choice"] == "required"
        return {
            "status": "completed",
            "id": f"resp_{index}",
            "output": [{
                "type": "function_call",
                "name": names[index],
                "call_id": f"call_{index}",
                "arguments": "{}",
            }],
        }
    assert body["tool_choice"] == "none"
    return {
        "status": "completed",
        "id": "resp_final",
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": json.dumps({
                    "explanation": "Пилоты проверили гипотезы, а расчёт учёл лимиты.",
                    "recommendation": "Проверить кампанию перед запуском.",
                }, ensure_ascii=False),
            }],
        }],
    }


class OpenAIDemoAgentTest(unittest.TestCase):
    def test_main_button_displays_actual_ai_response(self) -> None:
        from streamlit.testing.v1 import AppTest

        submitted: list[dict] = []
        explanation = "Выбор модели: сегменты <проверены>, бюджет учтён."
        recommendation = "Рекомендация модели: согласуйте выбранные кампании."

        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            response = _api_response(body, len(submitted))
            submitted.append(body)
            if len(submitted) == 4:
                response["output"][0]["content"][0]["text"] = json.dumps({
                    "explanation": explanation,
                    "recommendation": recommendation,
                }, ensure_ascii=False)
            return io.BytesIO(json.dumps(response).encode("utf-8"))

        with (
            patch("openai_demo_agent.require_api_key", return_value="test-key"),
            patch("openai_demo_agent.urlopen", side_effect=fake_urlopen),
        ):
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=30)
            self.assertEqual(len(submitted), 0)
            next(button for button in app.button
                 if button.label == "Сформировать оптимальный план").click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(submitted), 4)
        self.assertTrue(app.session_state["current_ai"]["active"])
        panel = next(item.value for item in app.markdown
                     if 'class="ai-report"' in item.value)
        self.assertIn(html.escape(explanation), panel)
        self.assertIn(html.escape(recommendation), panel)
        self.assertNotIn("<проверены>", panel)

    def test_real_tool_protocol_only_sends_aggregates(self) -> None:
        local = _local_result()
        submitted: list[dict] = []
        local_runs: list[int] = []

        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            submitted.append(body)
            response = _api_response(body, len(submitted) - 1)
            return io.BytesIO(json.dumps(response).encode("utf-8"))

        def fake_run(budget, seed=42, on_event=None):
            local_runs.append(budget)
            return local

        overview = {
            "segment_map": [
                {"arpu_segment": "large", "audience_count": 120},
                {"arpu_segment": "large", "audience_count": 180},
                {"arpu_segment": "rare", "audience_count": 1},
            ],
            "audience_count": 301,
            "tariff_count": 2,
            "max_budget": 100_000,
            "max_contacts": 15_000,
        }
        with (
            patch("openai_demo_agent.require_api_key", return_value="test-key"),
            patch("openai_demo_agent.urlopen", side_effect=fake_urlopen),
            patch("openai_demo_agent.run_scenario", side_effect=fake_run),
            patch("openai_demo_agent.get_overview", return_value=overview),
        ):
            result, info = run_openai_demo(1_000)

        self.assertIs(result, local)
        self.assertEqual(local_runs, [1_000])
        self.assertTrue(info["active"])
        self.assertEqual(info["tools_called"], [
            "check_hypotheses", "analyze_segments", "calculate_campaigns",
        ])
        self.assertEqual(info["response_id"], "resp_final")
        self.assertEqual(len(submitted), 4)
        self.assertTrue(all(body["store"] is False for body in submitted))
        serialized = json.dumps(submitted, ensure_ascii=False)
        for forbidden in ("must-stay-local", "ID_NUMBER", "mock_result", '"rare"'):
            self.assertNotIn(forbidden, serialized)

    def test_missing_key_uses_local_result(self) -> None:
        local = _local_result()
        with (
            patch("openai_demo_agent.require_api_key", side_effect=RuntimeError("missing")),
            patch("openai_demo_agent.run_scenario", return_value=local) as run,
            patch("openai_demo_agent.urlopen") as request,
        ):
            result, info = run_openai_demo(1_000)
        self.assertIs(result, local)
        self.assertFalse(info["active"])
        run.assert_called_once()
        request.assert_not_called()

    def test_network_error_uses_local_result(self) -> None:
        local = _local_result()
        with (
            patch("openai_demo_agent.require_api_key", return_value="test-key"),
            patch("openai_demo_agent.urlopen", side_effect=URLError("offline")),
            patch("openai_demo_agent.run_scenario", return_value=local) as run,
        ):
            result, info = run_openai_demo(1_000)
        self.assertIs(result, local)
        self.assertFalse(info["active"])
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
