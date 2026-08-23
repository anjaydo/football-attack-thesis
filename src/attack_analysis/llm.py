from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .agent_tools import EvidenceToolRegistry, verify_verdict
from .research import (
    EvidenceBundle,
    EvidenceClaim,
    EvaluationVerdict,
    insufficient_evidence_verdict,
)


PROMPT_VERSION = "attack-verdict-v1"


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
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'llm' extra to use OpenAI Responses API") from error
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instructions = load_project_skill_instructions(project_root)
        self.max_tool_rounds = max_tool_rounds

    def evaluate(self, bundle: EvidenceBundle, tools: EvidenceToolRegistry) -> EvaluationVerdict:
        if not bundle.downstream_ready:
            return insufficient_evidence_verdict(bundle.attack.attack_id, bundle.limitations, model=self.model)
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "input": (
                "Evaluate this attack only from the structured bundle. Call verify_claims before the final answer.\n"
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
                instructions=self.instructions,
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
        if last_verification["arguments"].get("claims") != [claim.to_dict() for claim in verdict.claims]:
            raise ValueError("final claims differ from the claims passed to verify_claims")
        return verdict
