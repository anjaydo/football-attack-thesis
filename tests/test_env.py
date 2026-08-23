from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from attack_analysis.env import load_project_environment, openai_api_key_configured


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
