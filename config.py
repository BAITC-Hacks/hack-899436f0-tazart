"""Load local API credentials without putting them in Git or logs."""

import os
from pathlib import Path


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_KEY_NAMES = {"openai": "OPENAI_API_KEY", "nvidia": "NVIDIA_API_KEY"}


def load_local_env(path: Path | None = None) -> None:
    """Read the two keys from a simple KEY=value .env file, if present.

Already-exported environment variables take precedence. The file stays local
and this function never prints credential values.
    """
    env_path = path if path is not None else Path(__file__).with_name(".env")
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in _KEY_NAMES.values():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            os.environ.setdefault(name, value)


def require_api_key(provider: str) -> str:
    """Return a configured key or fail before making a provider API request."""
    try:
        name = _KEY_NAMES[provider.lower()]
    except KeyError as exc:
        raise ValueError("provider must be 'openai' or 'nvidia'") from exc
    load_local_env()
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(f"{name} is missing; set it in the environment or local .env")
    return key
