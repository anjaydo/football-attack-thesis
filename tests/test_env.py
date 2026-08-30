from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from attack_analysis.env import (
    DEFAULT_OPENAI_MODEL,
    load_project_environment,
    openai_api_key_configured,
    openai_model,
)


class EnvironmentLoadingTests(unittest.TestCase):
    def test_loads_project_env_without_overriding_existing_key(self) -> None:
        original = os.environ.get("OPENAI_API_KEY")
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ".env").write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
                os.environ["OPENAI_API_KEY"] = "from-process"
                self.assertTrue(load_project_environment(root))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-process")
                self.assertTrue(openai_api_key_configured())
        finally:
            if original is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original

    def test_openai_model_defaults_to_allowed_nano(self) -> None:
        original = os.environ.get("OPENAI_MODEL")
        try:
            os.environ.pop("OPENAI_MODEL", None)
            self.assertEqual(openai_model(), DEFAULT_OPENAI_MODEL)
            self.assertEqual(DEFAULT_OPENAI_MODEL, "gpt-5.4-nano")
            os.environ["OPENAI_MODEL"] = "gpt-5.4-mini"
            self.assertEqual(openai_model(), "gpt-5.4-mini")
        finally:
            if original is None:
                os.environ.pop("OPENAI_MODEL", None)
            else:
                os.environ["OPENAI_MODEL"] = original


class EnvironmentLoadingTests(unittest.TestCase):
    def test_loads_project_env_without_overriding_existing_key(self) -> None:
        original = os.environ.get("OPENAI_API_KEY")
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ".env").write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
                os.environ["OPENAI_API_KEY"] = "from-process"
                self.assertTrue(load_project_environment(root))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-process")
                self.assertTrue(openai_api_key_configured())
        finally:
            if original is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original


if __name__ == "__main__":
    unittest.main()
