from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent_tools import EvidenceToolRegistry, verify_verdict
from .env import load_project_environment
from .research import (
    AttackConclusionV2,
    EvidenceBundle,
    EvidenceClaim,
    EvaluationVerdict,
    insufficient_evidence_verdict,
)


PROMPT_VERSION = "attack-verdict-v1"
PROMPT_VERSION_V2 = "attack-verdict-v2"


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


def conclusion_v2_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "attack_id": {"type": "string"},
            "chance_quality": {
                "type": "string",
                "enum": ["high", "medium", "low", "unknown"],
            },
            "xg_probability": {"type": ["number", "null"]},
            "tactical_read": {"type": "string"},
            "structural_features": {"type": "array", "items": {"type": "string"}},
            "comparison_to_role_benchmark": {"type": ["string", "null"]},
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
            "abstained_effectiveness": {"type": "boolean"},
        },
        "required": [
            "attack_id",
            "chance_quality",
            "xg_probability",
            "tactical_read",
            "structural_features",
            "comparison_to_role_benchmark",
            "claims",
            "limitations",
            "abstained_effectiveness",
        ],
    }


def _chance_quality_from_xg(xg: float | None) -> str:
    if xg is None:
        return "unknown"
    if xg >= 0.25:
        return "high"
    if xg >= 0.10:
        return "medium"
    return "low"


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

    def evaluate_v2(self, bundle: EvidenceBundle, tools: EvidenceToolRegistry) -> AttackConclusionV2:
        xg = bundle.attack.xg_sum if bundle.quality_gates.get("xg_scored") else None
        chance = _chance_quality_from_xg(xg)
        evidence_id = bundle.attack.evidence_ids[-1] if bundle.attack.evidence_ids else bundle.attack.attack_id
        features = [
            f"progression {bundle.attack.progression_m:.1f} m",
            f"events {bundle.attack.event_count}",
            "entered final third" if bundle.attack.entered_final_third else "did not enter final third",
        ]
        if bundle.attack.entered_penalty_box:
            features.append("entered penalty box")
        spatial = bundle.spatial_context or {}
        xg_raw = bundle.xg_raw or {}
        gk_distance = spatial.get("gk_distance_m")
        if gk_distance is not None:
            features.append(f"GK {float(gk_distance):.1f} m from shooter")
        if xg_raw.get("body_part"):
            features.append(f"body_part {xg_raw['body_part']}")
        if spatial.get("defenders_in_box") is not None:
            features.append(f"{int(spatial['defenders_in_box'])} defenders in the box")
        if spatial.get("foes_in_shot_corridor") is not None:
            features.append(f"{int(spatial['foes_in_shot_corridor'])} foes on the shot line")
        if xg_raw.get("play_pattern"):
            features.append(f"play_pattern {xg_raw['play_pattern']}")
        buildup = dict(spatial.get("possession_buildup") or {})
        if buildup.get("assist_type"):
            assist = dict(buildup.get("assist") or {})
            features.append(
                f"last {buildup['assist_type']} {float(assist.get('dt_sec') or 0):.1f}s before terminal"
            )
        if buildup.get("cross_then_header"):
            features.append("cross then headed attempt")
        comparison = None
        if bundle.benchmarks:
            first = bundle.benchmarks[0]
            comparison = (
                f"{first.role} {first.metric} {first.value_per90:.2f}/90 "
                f"(n={first.sample_size}, {first.season})"
            )
        xg_text = "unknown" if xg is None else f"{xg:.3f} (research_only)"
        spatial_bits = []
        if gk_distance is not None:
            spatial_bits.append(f"thủ môn cách {float(gk_distance):.1f} m")
        if spatial.get("defenders_in_box") is not None:
            spatial_bits.append(f"{int(spatial['defenders_in_box'])} đối phương trong vòng cấm")
        if buildup.get("assist_type"):
            assist = dict(buildup.get("assist") or {})
            spatial_bits.append(
                f"{buildup['assist_type']} {float(assist.get('dt_sec') or 0):.1f}s trước dứt điểm"
            )
        spatial_clause = f" Freeze-frame: {', '.join(spatial_bits)}." if spatial_bits else ""
        if bundle.attack.goals:
            spatial_clause += " Cú dứt điểm thành bàn."
        claims = (
            EvidenceClaim(
                text=f"Chance quality {chance}; xG-v2 {xg_text}.",
                evidence_ids=(evidence_id,),
            ),
        )
        conclusion = AttackConclusionV2(
            attack_id=bundle.attack.attack_id,
            chance_quality=chance,
            xg_probability=xg,
            tactical_read=(
                f"Chuỗi {bundle.attack.duration_sec:.1f}s tiến {bundle.attack.progression_m:.1f} m "
                f"với {bundle.attack.shots} dứt điểm.{spatial_clause} "
                "Đây không phải nhãn hiệu quả đã xác thực."
            ),
            structural_features=tuple(features),
            comparison_to_role_benchmark=comparison,
            claims=claims,
            limitations=bundle.limitations,
            model="deterministic",
            prompt_version=PROMPT_VERSION_V2,
        )
        conclusion.validate(bundle.evidence_ids())
        return conclusion


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

    def evaluate_v2(self, bundle: EvidenceBundle, tools: EvidenceToolRegistry) -> AttackConclusionV2:
        response_language = getattr(self, "response_language", "English")
        response_instructions = (
            self.instructions
            + "\n\n# Output language\n"
            + f"All user-visible claim text, tactical_read and limitations MUST be written in {response_language}. "
            + "Keep metric names and identifiers in their original form.\n"
            + "Do not emit a categorical effectiveness label. Set abstained_effectiveness=true. "
            + "Cite only evidence_ids from the bundle (action_id, xg record, benchmark_id). "
            + "Web search is context only; never use it for numeric xG or benchmark numbers."
        )
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": response_instructions,
            "input": (
                "Write an Attack-v2 research conclusion from this evidence bundle. "
                "Call verify_claims before the final answer. "
                "After verify_claims succeeds, copy the verified claims into the final JSON. "
                f"Write in {response_language}.\n"
                + json.dumps(bundle.to_dict(), ensure_ascii=False)
            ),
            "tools": tools.schemas(),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "attack_conclusion_v2",
                    "strict": True,
                    "schema": conclusion_v2_json_schema(),
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
        payload = json.loads(response.output_text)
        payload.update(
            {
                "model": self.model,
                "prompt_version": PROMPT_VERSION_V2,
                "tool_trace": tools.trace,
                "effectiveness_verdict": None,
            }
        )
        conclusion = AttackConclusionV2.from_dict(payload)
        conclusion.validate(bundle.evidence_ids())
        verification_calls = [item for item in tools.trace if item.get("tool") == "verify_claims"]
        if not verification_calls:
            raise ValueError("LLM must call verify_claims before returning a conclusion")
        last_verification = verification_calls[-1]
        if not last_verification["result"].get("valid"):
            raise ValueError("LLM returned claims after a failed verification")
        return replace(conclusion, tool_trace=tuple(dict(item) for item in tools.trace))
