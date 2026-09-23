"""One minimal OpenAI connection check; never sends project data."""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import require_api_key


def main() -> int:
    try:
        api_key = require_api_key("openai")
        body = json.dumps({
            "model": "gpt-4.1-nano",
            "input": "Ответь ровно одним словом: готово.",
            "max_output_tokens": 32,
            "store": False,
        }).encode("utf-8")
        request = Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        content = (
            part.get("text", "")
            for item in payload.get("output", [])
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        if payload.get("status") != "completed" or not any(content):
            print("Проверка OpenAI API: ответ не завершён.", file=sys.stderr)
            return 1
    except HTTPError as exc:
        print(f"Проверка OpenAI API: ошибка HTTP {exc.code}.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError):
        print("Проверка OpenAI API: соединение недоступно.", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, json.JSONDecodeError):
        print("Проверка OpenAI API: ошибка конфигурации или ответа.", file=sys.stderr)
        return 1
    print("Проверка OpenAI API: успешно.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
