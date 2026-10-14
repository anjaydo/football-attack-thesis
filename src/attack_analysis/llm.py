from __future__ import annotations

import json
import re
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
PROMPT_VERSION_V2 = "attack-verdict-v2.3-analyst-voice"


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


_ZONE_VI = {
    "defensive_third": "1/3 sân nhà của đội tấn công",
    "middle_third": "khu giữa sân",
    "final_third": "1/3 cuối sân",
    "penalty_box": "vòng cấm",
}
_CHANNEL_VI = {"central": "trung lộ", "half_space": "hành lang trong", "wide": "biên"}
_OUTCOME_VI = {"goal": "thành bàn", "no_goal": "không thành bàn"}
_ACTION_VI = {
    "pass": "chuyền",
    "drive": "dẫn bóng",
    "cross": "tạt bóng",
    "shot": "sút",
    "throw_in": "ném biên",
    "shot_attempt": "dứt điểm",
    "headed_attempt": "đánh đầu dứt điểm",
    "header": "đánh đầu",
    "tackle": "tắc bóng",
    "block": "chặn bóng",
    "forced_goal_line_out": "buộc đối phương phá ra biên ngang",
}


_FIELD_VI = {
    "left_censored": "không quan sát được đoạn đầu",
    "complete": "quan sát trọn vẹn",
    "sequence_status": "trạng thái chuỗi",
    "largest_forward_gain": "bước tiến lớn nhất",
    "forward_gain_m": "mức tiến",
    "penalty_box_entry": "điểm vào vòng cấm",
    "entered_penalty_box": "vào vòng cấm",
    "opponent_touches": "số lần đối phương chạm bóng",
    "ended_in_turnover": "mất bóng",
    "shot_outcome_source": "nguồn kết quả",
    "shot_outcome": "kết quả",
    "body_part": "bộ phận dứt điểm",
    "no_goal": "không thành bàn",
    "xg_max": "xG cao nhất",
    "xg_sum": "tổng xG",
    "key_moments": "các mốc chính",
    "first_entry": "lần đầu vào",
    "first entry": "lần đầu vào",
    "terminal": "pha kết thúc",
    "xG_sum": "tổng xG",
    "xG_max": "xG cao nhất",
    "xG-v2": "xG",
    "XG-v2": "xG",
    "half-space": "hành lang trong",
    "video_scoreboard": "bảng tỉ số trên video",
    "video scoreboard": "bảng tỉ số trên video",
    "event_restart": "giao bóng lại sau bàn thắng",
    "goal": "bàn thắng",
}
_VIETNAMESE = {"vietnamese", "tiếng việt", "vi"}
_LONG_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d{4,}")
_EVIDENCE_TOKEN_PATTERN = r"(?:xg:)?(?:actv2|termv2|atkv2|xgv2)_[0-9a-f]+|footpass-hdf5:[\w:.-]+"
_EVIDENCE_PAREN = re.compile(rf"\s*\((?:evidence[_ ]ids?\s*:?\s*)?[^()]*?(?:{_EVIDENCE_TOKEN_PATTERN})[^()]*\)")
_EVIDENCE_TOKEN = re.compile(rf"\s*\b(?:{_EVIDENCE_TOKEN_PATTERN})")
_KEY_VALUE_PAREN = re.compile(r"\s*\([^()]*\b[a-z_/]+=(?:true|false)\b[^()]*\)", re.IGNORECASE)
_T_EQUALS = re.compile(r"\bt\s*=\s*(\d+(?:\.\d+)?)\s*(?:giây|s)?(?!\w)")
_DECIMAL_SECOND = re.compile(r"\bgiây(?: thứ)?\s+(\d+[.,]\d+)")
_LABEL_PAREN = re.compile(
    r"\s*\((?:pha kết thúc|bước tiến lớn nhất|(?:kết quả:\s*)?(?:bàn thắng|không thành bàn)"
    r"|\d+(?:[.,]\d+)?\s+trong dữ liệu)\)|\s+trong dữ liệu là \d+(?:[.,]\d+)?",
    re.IGNORECASE,
)
_DECIMAL_METRES = re.compile(r"\b(\d+[.,]\d+)\s*m\b")
_REPEATED_WORDS = re.compile(r"\b((?:\w+\s+){0,2}\w+)(?:\s+\1\b)+")
_SNAKE_ASSIGNMENT = re.compile(r"\b([a-z]+(?:_[a-z]+)+)\s*=\s*")
_REPEATED_PAREN = re.compile(r"\b([^(),;]{3,}?)\s*\(\1\b([^()]*)\)", re.IGNORECASE)


def _vietnamese_terms() -> dict[str, str]:
    return {**_FIELD_VI, **_ZONE_VI, **_CHANNEL_VI, **_ACTION_VI}


def _glossary(language: str) -> str:
    if language.casefold() not in _VIETNAMESE:
        return ""
    terms = _vietnamese_terms()
    return "- Vietnamese terms: " + "; ".join(f"{key} = {value}" for key, value in terms.items()) + "\n"


def _polish_text(text: str, language: str) -> str:
    """Strip leaked evidence IDs, round decimals and translate schema tokens in user-visible prose."""

    text = _EVIDENCE_PAREN.sub("", text)
    text = _EVIDENCE_TOKEN.sub("", text)
    text = _KEY_VALUE_PAREN.sub("", text)
    text = _LONG_DECIMAL.sub(lambda match: f"{float(match.group(0)):.3f}", text)
    if language.casefold() not in _VIETNAMESE:
        return text
    text = _T_EQUALS.sub(lambda match: f"giây thứ {int(float(match.group(1)) + 0.5)}", text)
    text = _DECIMAL_SECOND.sub(
        lambda match: f"giây thứ {int(float(match.group(1).replace(',', '.')) + 0.5)}", text
    )
    text = _DECIMAL_METRES.sub(lambda match: f"{int(float(match.group(1).replace(',', '.')) + 0.5)} m", text)
    text = _SNAKE_ASSIGNMENT.sub(lambda match: f"{match.group(1)}: ", text)
    for token in sorted(_vietnamese_terms(), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(token)}\b", _vietnamese_terms()[token], text)
    text = _REPEATED_PAREN.sub(lambda match: match.group(1) + match.group(2), text)
    text = _LABEL_PAREN.sub("", text)
    return _REPEATED_WORDS.sub(lambda match: match.group(1), text)


_ZONE_AREA_VI = {
    "defensive_third": "phần sân nhà",
    "middle_third": "khu trung tuyến",
    "final_third": "1/3 cuối sân",
    "penalty_box": "vòng cấm",
}
_CHANNEL_AREA_VI = {"central": "trung lộ", "half_space": "hành lang trong", "wide": "hành lang biên"}
_TERMINAL_NOUN_VI = {"shot_attempt": "cú dứt điểm", "headed_attempt": "pha đánh đầu"}
_OUTCOME_SOURCE_VI = {
    "video_scoreboard": " (xác nhận qua bảng tỉ số)",
    "event_restart": " (xác nhận qua pha giao bóng lại)",
}


def _xg_band_vi(xg: float) -> str:
    if xg >= 0.25:
        return "cơ hội lớn"
    if xg >= 0.10:
        return "nửa cơ hội"
    return "cơ hội khó"


def _timeline_read(context: dict[str, Any], xg: float | None = None) -> str:
    """Deterministic Vietnamese narrative in the match-analyst voice described in the skill."""

    events = [item for item in context.get("events") or () if isinstance(item, dict)]
    if not events:
        return ""
    by_step = {int(item["step"]): item for item in events}
    terminal = next((item for item in events if item.get("terminal")), events[-1])
    terminal_step = int(terminal["step"])
    outcome = str(context.get("shot_outcome") or "unknown")

    def who(item: dict[str, Any]) -> str:
        return f"số {item['jersey']}" if item.get("jersey") is not None else "đội tấn công"

    def verb(item: dict[str, Any]) -> str:
        return _ACTION_VI.get(str(item.get("action")), str(item.get("action")))

    def place(item: dict[str, Any]) -> str:
        zone = _ZONE_AREA_VI.get(str(item.get("zone")), "")
        channel = _CHANNEL_AREA_VI.get(str(item.get("channel")), "")
        return ", ".join(value for value in (zone, channel) if value) or "vị trí chưa rõ"

    def at(item: dict[str, Any]) -> str:
        return f"Ở giây thứ {float(item.get('t_sec') or 0):.0f}"

    moments: dict[int, set[str]] = {}
    gains: dict[int, float] = {}
    for moment in context.get("key_moments") or ():
        step = int(moment["step"])
        moments.setdefault(step, set()).add(str(moment["moment"]))
        if moment.get("forward_gain_m") is not None:
            gains[step] = float(moment["forward_gain_m"])

    first = events[0]
    lead = "Bàn thắng" if outcome == "goal" else (
        "Cú dứt điểm" if terminal.get("action") in _TERMINAL_NOUN_VI else "Đợt tấn công"
    )
    gain_step = next(iter(gains), None)
    route = (
        f" qua {_CHANNEL_AREA_VI[str(by_step[gain_step]['channel'])]}"
        if gain_step is not None and str(by_step[gain_step].get("channel")) in _CHANNEL_AREA_VI
        else ""
    )
    ending = _TERMINAL_NOUN_VI.get(str(terminal.get("action")), f"pha {verb(terminal)}")
    parts = [
        f"{lead} đến từ một đợt lên bóng kéo dài khoảng {float(context.get('duration_sec') or 0):.0f} giây, "
        f"đi từ {_ZONE_AREA_VI.get(str(first.get('zone')), 'vị trí chưa rõ')}{route} "
        f"rồi kết thúc bằng {ending} ở {place(terminal)}."
    ]
    if context.get("sequence_status") == "left_censored":
        parts.append(
            "Đoạn triển khai trước đó không có trong dữ liệu; diễn biến quan sát được bắt đầu khi "
            f"{who(first)} {verb(first)} ở {place(first)}."
        )
    else:
        parts.append(f"Đợt bóng bắt đầu khi {who(first)} {verb(first)} ở {place(first)}.")

    last_told = int(first["step"])
    for step in sorted(moments):
        item = by_step.get(step)
        kinds = moments[step]
        if item is None or step == last_told or step >= terminal_step:
            continue
        if item.get("team") == "opponent":
            parts.append(f"{at(item)}, đối phương có một pha {verb(item)} nhưng không giải toả được bóng.")
            last_told = step
            continue
        linking = sum(1 for other in events if last_told < int(other["step"]) < step and other.get("team") == "attacking")
        prefix = f"Sau {linking} nhịp chạm nối tiếp, ở" if linking >= 2 else at(item)
        if not prefix.startswith("Ở"):
            sentence = f"{prefix} giây thứ {float(item.get('t_sec') or 0):.0f}, {who(item)} {verb(item)}"
        else:
            sentence = f"{prefix}, {who(item)} {verb(item)}"
        if "cross" in kinds:
            sentence += f" từ {_CHANNEL_AREA_VI.get(str(item.get('channel')), 'biên')}"
        entries = [kind for kind in kinds if kind.startswith("first_entry_")]
        if entries:
            zone = entries[0].removeprefix("first_entry_")
            if item.get("action") in {"shot", "header"}:
                sentence += f" ngay trong {_ZONE_AREA_VI.get(zone, zone)}"
            elif zone == "penalty_box":
                sentence += " đưa bóng vào vòng cấm"
            else:
                sentence += f" đưa bóng lên {_ZONE_AREA_VI.get(zone, zone)}"
        if step in gains:
            sentence += f" — quãng tiến dài nhất của cả đợt, khoảng {gains[step]:.0f} m"
        parts.append(sentence + ".")
        last_told = step

    finish = f"{at(terminal)}, {who(terminal)} {verb(terminal)} ở {place(terminal)}"
    if "first_entry_penalty_box" in moments.get(terminal_step, set()):
        finish += " ngay trong lần đầu bóng vào vòng cấm"
    parts.append(finish + ".")

    source = _OUTCOME_SOURCE_VI.get(str(context.get("shot_outcome_source")), "")
    if outcome == "goal":
        parts.append(f"Bóng đi vào lưới{source}.")
    elif outcome == "no_goal":
        parts.append(f"Cú dứt điểm không thành bàn{source}.")
    if xg is not None:
        sentence = f"Theo mô hình xG, đây là {_xg_band_vi(xg)} (xG {xg:.2f})"
        if outcome == "goal" and xg < 0.25:
            sentence += " — pha dứt điểm vượt kỳ vọng"
        elif outcome == "no_goal" and xg >= 0.25:
            sentence += " — một cơ hội lớn bị bỏ lỡ"
        parts.append(sentence + ".")

    if gain_step is not None and gains[gain_step] >= 20 and gain_step in by_step:
        item = by_step[gain_step]
        parts.append(
            f"Điểm mấu chốt là pha {verb(item)} của {who(item)} ở giây thứ {float(item.get('t_sec') or 0):.0f}: "
            f"một nhịp chạm cho quãng tiến khoảng {gains[gain_step]:.0f} m, kéo cả đợt bóng lên "
            f"{_ZONE_AREA_VI.get(str(item.get('zone')), 'phía trên')}."
        )
    elif any("cross" in kinds for kinds in moments.values()):
        step = next(step for step, kinds in sorted(moments.items()) if "cross" in kinds)
        parts.append(
            f"Điểm mấu chốt là quả tạt ở giây thứ {float(by_step[step].get('t_sec') or 0):.0f}, "
            "đưa bóng thẳng vào khu vực dứt điểm."
        )
    elif len(events) <= 3:
        parts.append(f"Đợt bóng ngắn và trực diện: chỉ {len(events)} nhịp chạm từ lúc quan sát đến khi dứt điểm.")
    return " ".join(parts)


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
            spatial_bits.append(f"thủ môn đứng cách người sút {float(gk_distance):.0f} m")
        if spatial.get("defenders_in_box") is not None:
            spatial_bits.append(f"có {int(spatial['defenders_in_box'])} cầu thủ đối phương trong vòng cấm")
        if buildup.get("assist_type"):
            assist = dict(buildup.get("assist") or {})
            delivery = {"cross": "quả tạt", "pass": "đường chuyền cuối"}.get(
                str(buildup["assist_type"]), str(buildup["assist_type"])
            )
            spatial_bits.append(f"{delivery} đến {float(assist.get('dt_sec') or 0):.1f} giây trước cú sút")
        spatial_clause = (
            f" Ở khoảnh khắc dứt điểm, {', '.join(spatial_bits)}." if spatial_bits else ""
        )
        timeline = _timeline_read(bundle.sequence_context or {}, xg)
        if bundle.attack.goals and "vào lưới" not in timeline:
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
                (
                    f"{timeline}{spatial_clause} "
                    if timeline
                    else (
                        f"Chuỗi {bundle.attack.duration_sec:.1f}s tiến {bundle.attack.progression_m:.1f} m "
                        f"với {bundle.attack.shots} dứt điểm.{spatial_clause} "
                    )
                )
                + "Đây không phải nhãn hiệu quả đã xác thực."
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
            + "Web search is context only; never use it for numeric xG or benchmark numbers.\n\n"
            + "# Attack-v2 conclusion\n"
            + "- Build tactical_read from sequence_context.events in the 'Analyst voice' of the skill: one "
            + "opening sentence naming the mechanism, then the move as cause and effect through the phases, "
            + "then the chance quality, the real outcome and one closing tactical takeaway. 5-7 sentences.\n"
            + "- sequence_context.key_moments is authoritative for which event is the first entry into each zone, "
            + "the largest forward gain, crosses, opponent touches and the terminal. Never move these moments "
            + "to other events; events with no key moment are connecting play and get one short clause.\n"
            + "- Locate moments by time from the first event in whole seconds (t_sec rounded, e.g. 'ở giây thứ 9') "
            + "and shirt number (jersey, e.g. 'số 7'), not by step numbers. Omit the shirt number when jersey "
            + "is missing.\n"
            + "- Describe only what the events and freeze-frame show. Do not describe defenders' reactions, "
            + "pressing, marking, transitions/counter-attacks or player intentions unless an event or "
            + "freeze-frame field shows them. Leave out opponent_touch_after_terminal: it is not part of the attack.\n"
            + "- structural_features: 4-6 short analyst notes (pattern, progression route, final-third entry, "
            + "delivery, finishing location, outcome), each a plain phrase rather than 'field: value'.\n"
            + "- Write one claim per key moment (progression, box entry, delivery, finish, xG, outcome) as a "
            + "full analyst sentence, each citing the evidence_id of that event or record.\n"
            + "- If shot_outcome is goal or no_goal, state it plainly; if unknown, say the outcome is not known.\n"
            + "- Do not recompute distances or probabilities. Show xG with 2 decimals and metres as whole "
            + "numbers ('gần 30 m').\n"
            + f"- Write natural {response_language} for a football audience: translate zone/channel/action "
            + "tokens and never put evidence_ids, field names or snake_case tokens in the text. Evidence IDs "
            + "belong only in the evidence_ids arrays.\n"
            + _glossary(response_language)
        )
        conclusion_tools = [item for item in tools.schemas() if item.get("name") != "get_attack_evidence"]
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": response_instructions,
            "input": (
                "Write an Attack-v2 research conclusion from this evidence bundle. "
                "Read sequence_context.events step by step first, then the xG and freeze-frame evidence. "
                "The complete bundle is included below, so get_attack_evidence is not needed. "
                "Call verify_claims once with all claims; only if it reports invalid claims, fix them and "
                "verify again. "
                "After verify_claims succeeds, copy the verified claims into the final JSON. "
                f"Write in {response_language}.\n"
                + json.dumps(bundle.to_dict(), ensure_ascii=False)
            ),
            "tools": conclusion_tools,
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
            rounds += 1
            response = self.client.responses.create(
                model=self.model,
                instructions=response_instructions,
                input=response_items + tool_outputs,
                tools=conclusion_tools,
                tool_choice="none" if rounds >= self.max_tool_rounds else "auto",
                text=request["text"],
                store=False,
            )
        payload = json.loads(response.output_text)
        payload["tactical_read"] = _polish_text(str(payload.get("tactical_read") or ""), response_language)
        payload["structural_features"] = [
            _polish_text(str(item), response_language) for item in payload.get("structural_features") or ()
        ]
        for claim in payload.get("claims") or ():
            claim["text"] = _polish_text(str(claim.get("text") or ""), response_language)
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
