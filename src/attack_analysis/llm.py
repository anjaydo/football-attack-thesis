from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent_tools import EvidenceToolRegistry, verify_verdict
from .env import load_project_environment
from .research import (
    EvidenceBundle,
    EvidenceClaim,
    EvaluationVerdict,
    insufficient_evidence_verdict,
)


PROMPT_VERSION = "attack-verdict-v1"


def describe_openai_error(error: Exception) -> str:
    """Return an actionable message without exposing request or credential details."""
    status_code = getattr(error, "status_code", None)
    error_code = getattr(error, "code", None)
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        error_code = error_code or body.get("code")
        nested = body.get("error")
        if isinstance(nested, dict):
            error_code = error_code or nested.get("code")

    normalized = f"{error_code or ''} {error}".lower()
    if status_code == 429 and "insufficient_quota" in normalized:
        return (
            "OpenAI accepted the API key, but this API project has no available quota. "
            "Add API credits or raise the project usage limit, then retry after billing updates."
        )
    if status_code == 429:
        return "OpenAI rate-limited this request. Wait briefly and try again."
    if status_code == 401:
        return "OpenAI rejected the API key. Check OPENAI_API_KEY and restart the demo."
    if status_code == 403:
        return "This OpenAI API project does not have permission to use the selected model."
    if status_code == 404:
        return "The selected OpenAI model was not found or is not available to this API project."
    if isinstance(error, ValueError):
        return (
            "OpenAI returned a verdict that did not satisfy the evidence contract, even after an automatic retry. "
            "The deterministic baseline is shown instead."
        )
    return f"OpenAI evaluation failed: {type(error).__name__}. Check the Streamlit terminal for details."


def load_project_skill_instructions(project_root: Path) -> str:
    names = ("attack-evidence-analyst", "football-benchmark-retriever", "football-claim-verifier")
    sections: list[str] = []
    for name in names:
        path = project_root / ".cursor" / "skills" / name / "SKILL.md"
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
    if len(sections) != len(names):
        missing = [name for name in names if not (project_root / ".cursor" / "skills" / name / "SKILL.md").exists()]
        raise FileNotFoundError(f"missing project skills: {missing}")
    return "\n\n---\n\n".join(sections)


def verdict_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "attack_id": {"type": "string"},
            "label": {
                "type": "string",
                "enum": ["effective", "promising", "ineffective", "insufficient_evidence"],
            },
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "evidence_ids"],
                },
            },
            "limitations": {"type": "array", "items": {"type": "string"}},
            "abstained": {"type": "boolean"},
        },
        "required": ["attack_id", "label", "score", "confidence", "claims", "limitations", "abstained"],
    }


class LLMClient(ABC):
    @abstractmethod
    def evaluate(self, bundle: EvidenceBundle, tools: EvidenceToolRegistry) -> EvaluationVerdict:
        raise NotImplementedError


class DeterministicEvidenceClient(LLMClient):
    """Offline baseline with the same evidence and abstention contract as the API client."""

    def evaluate(self, bundle: EvidenceBundle, tools: EvidenceToolRegistry) -> EvaluationVerdict:
        if not bundle.downstream_ready:
            return insufficient_evidence_verdict(bundle.attack.attack_id, bundle.limitations, model="deterministic")
        attack = bundle.attack
        score = 1
        if attack.entered_final_third:
            score += 1
        if attack.entered_penalty_box or attack.progression_ratio >= 0.5:
            score += 1
        if attack.shots:
            score += 1
        if attack.xg_sum >= 0.25 or attack.goals:
            score += 1
        score = min(5, score)
        label = "effective" if score >= 4 else "promising" if score == 3 else "ineffective"
        evidence_id = attack.evidence_ids[-1]
        claims = (
            EvidenceClaim(
                text=(
                    f"Attack progression was {attack.progression_m:.1f} m and total xG was {attack.xg_sum:.2f}."
                ),
                evidence_ids=(evidence_id,),
            ),
        )
        verdict = EvaluationVerdict(
            attack_id=attack.attack_id,
            label=label,  # type: ignore[arg-type]
            score=score,
            confidence=max(0.0, 1.0 - max(attack.uncertainty.values(), default=0.0)),
            claims=claims,
            limitations=bundle.limitations,
            model="deterministic",
            prompt_version=PROMPT_VERSION,
        )
        verify_verdict(verdict, bundle)
        return verdict


class OpenAIResponsesClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        project_root: Path,
        api_key: str | None = None,
        max_tool_rounds: int = 6,
        max_contract_retries: int = 1,
        response_language: str = "English",
    ) -> None:
        load_project_environment(project_root)
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'llm' extra to use OpenAI Responses API") from error
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instructions = load_project_skill_instructions(project_root)
        self.max_tool_rounds = max_tool_rounds
        self.max_contract_retries = max_contract_retries
        self.response_language = response_language

    def evaluate(self, bundle: EvidenceBundle, tools: EvidenceToolRegistry) -> EvaluationVerdict:
        if not bundle.downstream_ready:
            return insufficient_evidence_verdict(bundle.attack.attack_id, bundle.limitations, model=self.model)
        max_retries = max(0, getattr(self, "max_contract_retries", 1))
        for retry_count in range(max_retries + 1):
            try:
                verdict = self._evaluate_once(bundle, tools)
            except ValueError as error:
                if retry_count >= max_retries:
                    raise
                tools.trace.append(
                    {
                        "event": "contract_retry",
                        "attempt": retry_count + 1,
                        "error_type": type(error).__name__,
                    }
                )
                continue
            return replace(
                verdict,
                usage={**verdict.usage, "contract_retries": retry_count},
            )
        raise AssertionError("unreachable")

    def _evaluate_once(self, bundle: EvidenceBundle, tools: EvidenceToolRegistry) -> EvaluationVerdict:
        response_language = getattr(self, "response_language", "English")
        response_instructions = (
            self.instructions
            + "\n\n# Output language\n"
            + f"All user-visible claim text and limitations MUST be written in {response_language}. "
            + "Keep only metric names, action labels, identifiers and source names in their original form."
        )
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": response_instructions,
            "input": (
                "Evaluate this attack only from the structured bundle. Call verify_claims before the final answer. "
                "After verify_claims succeeds, copy the verified claims into the final JSON without changing their "
                f"text or evidence_ids. Write claims and limitations in {response_language}.\n"
                + json.dumps(bundle.to_dict(), ensure_ascii=False)
            ),
            "tools": tools.schemas(),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "attack_evaluation_verdict",
                    "strict": True,
                    "schema": verdict_json_schema(),
                }
            },
            "store": False,
        }
        response = self.client.responses.create(**request)
        rounds = 0
        while rounds < self.max_tool_rounds:
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not calls:
                break
            tool_outputs: list[dict[str, Any]] = []
            for call in calls:
                arguments = json.loads(call.arguments)
                result = tools.call(call.name, arguments)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
            response_items = [
                item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
                for item in response.output
            ]
            response = self.client.responses.create(
                model=self.model,
                instructions=response_instructions,
                input=response_items + tool_outputs,
                tools=tools.schemas(),
                text=request["text"],
                store=False,
            )
            rounds += 1
        if rounds >= self.max_tool_rounds and any(
            getattr(item, "type", None) == "function_call" for item in response.output
        ):
            raise RuntimeError("LLM exceeded the maximum number of tool rounds")
        payload = json.loads(response.output_text)
        payload.update(
            {
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "tool_trace": tools.trace,
                "usage": {
                    "input_tokens": int(getattr(response.usage, "input_tokens", 0)),
                    "output_tokens": int(getattr(response.usage, "output_tokens", 0)),
                    "total_tokens": int(getattr(response.usage, "total_tokens", 0)),
                },
            }
        )
        verdict = EvaluationVerdict.from_dict(payload)
        verify_verdict(verdict, bundle)
        verification_calls = [item for item in tools.trace if item.get("tool") == "verify_claims"]
        if not verification_calls:
            raise ValueError("LLM must call verify_claims before returning a verdict")
        last_verification = verification_calls[-1]
        if not last_verification["result"].get("valid"):
            raise ValueError("LLM returned claims after a failed verification")
        final_claims = [claim.to_dict() for claim in verdict.claims]
        if last_verification["arguments"].get("claims") != final_claims:
            final_verification = tools.call(
                "verify_claims",
                {"attack_id": verdict.attack_id, "claims": final_claims},
            )
            if not final_verification.get("valid"):
                raise ValueError("final claims failed post-generation verification")
        return replace(verdict, tool_trace=tuple(dict(item) for item in tools.trace))
