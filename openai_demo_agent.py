"""OpenAI commentary around the existing deterministic campaign scenario.

Only compact, anonymous aggregates leave this process. The model chooses the
order of three read-only local tools; it never computes or changes the plan.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import require_api_key
from dashboard_service import get_overview, run_scenario


_LOGGER = logging.getLogger(__name__)
_RESPONSES_URL = "https://api.openai.com/v1/responses"
_MODEL = "gpt-4.1-nano"
_TOOL_NAMES = ("analyze_segments", "check_hypotheses", "calculate_campaigns")
_NETWORK_DEADLINE_SECONDS = 35.0
_REQUEST_TIMEOUT_SECONDS = 12.0

_TOOLS = [
    {
        "type": "function",
        "name": "analyze_segments",
        "description": "Вернуть обезличенную сводку размера аудитории и ARPU-сегментов.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "check_hypotheses",
        "description": "Локально проверить маркетинговые гипотезы по пилотам и вернуть только агрегированные итоги.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate_campaigns",
        "description": "Локально рассчитать итоговый портфель кампаний, лимиты и оценки; вернуть только агрегированные итоги.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
]

_INSTRUCTIONS = (
    "Ты аналитический агент для демонстрации тарифных кампаний Beeline. "
    "Выбери порядок вызова трёх локальных инструментов: analyze_segments, "
    "check_hypotheses, calculate_campaigns. Для заключения используй все три. "
    "Инструменты выполняют расчёты локально и возвращают только обезличенные агрегаты. "
    "Ты не изменяешь портфель, не пересчитываешь суммы и не называешь оценки фактическим эффектом. "
    "Не придумывай цифры, результаты скрытого зачёта или факты об отдельных абонентах. "
    "После инструментов кратко объясни выбор и дай одну практическую рекомендацию по-русски."
)

_FINAL_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "campaign_commentary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "recommendation": {"type": "string"},
            },
            "required": ["explanation", "recommendation"],
            "additionalProperties": False,
        },
    }
}


class _ServiceUnavailable(Exception):
    """An API or response-protocol error suitable for local fallback."""


def _post_response(api_key: str, body: dict, deadline: float) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _ServiceUnavailable
    request = Request(
        _RESPONSES_URL,
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=min(_REQUEST_TIMEOUT_SECONDS, remaining)) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, UnicodeError) as exc:
        raise _ServiceUnavailable from exc
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise _ServiceUnavailable
    if not isinstance(payload.get("output"), list):
        raise _ServiceUnavailable
    return payload


def _segment_summary() -> dict:
    overview = get_overview()
    by_arpu_segment: Counter[str] = Counter()
    for cell in overview["segment_map"]:
        by_arpu_segment[str(cell["arpu_segment"])] += int(cell["audience_count"])
    # Do not expose a tiny band's size indirectly as total minus visible bands.
    bands = (
        sorted(
            ({"arpu_segment": band[:32], "audience_count": count}
             for band, count in by_arpu_segment.items()),
            key=lambda item: (-item["audience_count"], item["arpu_segment"]),
        )
        if by_arpu_segment and min(by_arpu_segment.values()) >= 50 else []
    )
    return {
        "audience_count": int(overview["audience_count"]),
        "tariff_count": int(overview["tariff_count"]),
        "arpu_band_count": len(by_arpu_segment),
        "arpu_bands": bands,
        "budget_limit": int(overview["max_budget"]),
        "contact_limit": int(overview["max_contacts"]),
    }


def _hypothesis_summary(result: dict) -> dict:
    pilots = result["pilots"]
    positive = sum(float(pilot.get("observed_lift_ratio", 0)) > 0 for pilot in pilots)
    negative = sum(float(pilot.get("observed_lift_ratio", 0)) < 0 for pilot in pilots)
    totals = result["totals"]
    return {
        "pilots_run": int(totals["pilot_count"]),
        "pilots_with_positive_observed_lift": positive,
        "pilots_with_negative_observed_lift": negative,
        "pilot_contacts": int(totals["pilot_contacts"]),
        "pilot_cost": round(float(totals["pilot_cost"]), 2),
        "screened_out_count": len(result["rejected"]),
        "note": "Наблюдения пилотов зашумлены; они не равны гарантированному результату кампаний.",
    }


def _campaign_summary(result: dict) -> dict:
    totals = result["totals"]
    by_channel = Counter(str(campaign["channel"]) for campaign in result["campaigns"])
    estimated_net = result["agent_estimated_final_net"]
    return {
        "selected_campaigns": int(totals["campaign_count"]),
        "selected_channels": dict(sorted(by_channel.items())),
        "final_contacts": int(totals["final_contacts"]),
        "final_cost": round(float(totals["final_cost"]), 2),
        "budget_used_including_pilots": round(float(totals["budget_used"]), 2),
        "budget_remaining": round(float(totals["budget_remaining"]), 2),
        "contacts_remaining": int(totals["contacts_remaining"]),
        "estimated_final_net": round(float(estimated_net), 2) if estimated_net is not None else None,
        "estimate_note": "Ожидаемый чистый эффект — локальная оценка агента, не факт и не скрытый зачёт.",
    }


def _output_text(payload: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in payload["output"]
        if isinstance(item, dict) and item.get("type") == "message"
        for part in item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "output_text"
    )


def run_openai_demo(budget: int, seed: int = 42, on_event=None) -> tuple[dict, dict]:
    """Run the unchanged local scenario, with bounded OpenAI tool orchestration.

    Return ``(scenario, ai_info)``. ``ai_info`` has ``active``, ``explanation``,
    ``recommendation``, ``tools_called`` and ``response_id``. API failures use
    local mode; local calculation errors still propagate to the caller.
    """
    result: dict | None = None
    info = {
        "active": False,
        "explanation": "",
        "recommendation": "",
        "tools_called": [],
        "response_id": None,
    }

    def scenario() -> dict:
        nonlocal result
        if result is None:
            result = run_scenario(budget, seed=seed, on_event=on_event)
        return result

    try:
        api_key = require_api_key("openai")
    except RuntimeError:
        return scenario(), info

    deadline = time.monotonic() + _NETWORK_DEADLINE_SECONDS
    inputs: list[dict] = [{
        "role": "user",
        "content": (
            f"Сформируй пояснение оптимального плана при бюджете {budget} у. е. "
            "Сначала вызови все три локальных инструмента в нужном тебе порядке."
        ),
    }]
    called: list[str] = []
    try:
        for _ in _TOOL_NAMES:
            available = [tool for tool in _TOOLS if tool["name"] not in called]
            payload = _post_response(api_key, {
                "model": _MODEL,
                "instructions": _INSTRUCTIONS,
                "input": inputs,
                "tools": available,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "max_output_tokens": 160,
                "store": False,
            }, deadline)
            calls = [item for item in payload["output"]
                     if isinstance(item, dict) and item.get("type") == "function_call"]
            if len(calls) != 1:
                raise _ServiceUnavailable
            call = calls[0]
            name = call.get("name")
            call_id = call.get("call_id")
            if name not in _TOOL_NAMES or name in called or not isinstance(call_id, str) or not call_id:
                raise _ServiceUnavailable
            try:
                arguments = json.loads(call.get("arguments") or "{}")
            except (ValueError, TypeError) as exc:
                raise _ServiceUnavailable from exc
            if arguments != {}:
                raise _ServiceUnavailable

            if name == "analyze_segments":
                tool_result = _segment_summary()
            elif name == "check_hypotheses":
                tool_result = _hypothesis_summary(scenario())
            else:
                tool_result = _campaign_summary(scenario())

            inputs.extend(payload["output"])
            inputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(tool_result, ensure_ascii=False, allow_nan=False),
            })
            called.append(name)

        final = _post_response(api_key, {
            "model": _MODEL,
            "instructions": _INSTRUCTIONS,
            "input": inputs,
            "tools": _TOOLS,
            "tool_choice": "none",
            "parallel_tool_calls": False,
            "text": _FINAL_FORMAT,
            "max_output_tokens": 280,
            "store": False,
        }, deadline)
        try:
            commentary = json.loads(_output_text(final))
        except (TypeError, ValueError) as exc:
            raise _ServiceUnavailable from exc
        if not isinstance(commentary, dict):
            raise _ServiceUnavailable
        explanation = commentary.get("explanation")
        recommendation = commentary.get("recommendation")
        if not isinstance(explanation, str) or not isinstance(recommendation, str):
            raise _ServiceUnavailable
        explanation = explanation.strip()[:800]
        recommendation = recommendation.strip()[:500]
        if not explanation or not recommendation:
            raise _ServiceUnavailable
        info.update({
            "active": True,
            "explanation": explanation,
            "recommendation": recommendation,
            "tools_called": called,
            "response_id": final.get("id") if isinstance(final.get("id"), str) else None,
        })
        _LOGGER.info(
            "OpenAI Responses API completed; tools_called=%d; response_id=%s",
            len(called), info["response_id"],
        )
    except _ServiceUnavailable:
        _LOGGER.warning("OpenAI Responses API unavailable; using local calculation mode")

    return scenario(), info
