from __future__ import annotations

import os
from pathlib import Path


def load_project_environment(project_root: Path) -> bool:
    """Load local development secrets without overriding process-level configuration."""
    try:
        from dotenv import load_dotenv
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install python-dotenv or the demo/llm extra to load .env") from error
    env_path = project_root / ".env"
    return bool(load_dotenv(dotenv_path=env_path, override=False)) if env_path.exists() else False


def openai_api_key_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


# Cheapest allowed model on this project's OpenAI allowlist that still supports
# Responses API tool-calling for the conclusion step.
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"


def openai_model() -> str:
    configured = os.environ.get("OPENAI_MODEL", "").strip()
    return configured or DEFAULT_OPENAI_MODEL