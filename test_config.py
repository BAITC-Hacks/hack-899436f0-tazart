"""Credential loading works with both server environment and local .env."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config import require_api_key


class ConfigTest(unittest.TestCase):
    def test_server_environment_does_not_read_dotenv(self) -> None:
        for provider, name in (
            ("openai", "OPENAI_API_KEY"),
            ("nvidia", "NVIDIA_API_KEY"),
        ):
            with (
                self.subTest(provider=provider),
                patch.dict(os.environ, {name: "server-test-value"}, clear=True),
                patch("config.load_local_env", side_effect=PermissionError) as load,
            ):
                self.assertEqual(require_api_key(provider), "server-test-value")
                load.assert_not_called()

    def test_server_environment_without_dotenv(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"OPENAI_API_KEY": "server-test-value"}, clear=True),
            patch("config.__file__", str(Path(directory) / "config.py")),
        ):
            self.assertFalse((Path(directory) / ".env").exists())
            self.assertEqual(require_api_key("openai"), "server-test-value")

    def test_local_dotenv_when_environment_variable_is_absent(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch("config.__file__", str(Path(directory) / "config.py")),
        ):
            (Path(directory) / ".env").write_text(
                'OPENAI_API_KEY="local-test-value"\n', encoding="utf-8"
            )
            self.assertEqual(require_api_key("openai"), "local-test-value")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "local-test-value")

    def test_missing_key_raises_expected_error(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch("config.__file__", str(Path(directory) / "config.py")),
        ):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is missing"):
                require_api_key("openai")

    def test_empty_environment_value_keeps_precedence_over_dotenv(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True),
            patch("config.load_local_env") as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is missing"):
                require_api_key("openai")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
