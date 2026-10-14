from __future__ import annotations

import hashlib
import html
import importlib
import json
import logging
import math
from dataclasses import replace as dataclass_replace
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st

from attack_analysis.agent_tools import EvidenceToolRegistry
from attack_analysis.attack_evidence import build_attack_evidence, default_team_mapping
from attack_analysis.shot_xg import (
    ShotXGRequiredError,
    ending_event_action,
    is_shot_event,
    score_analysis_shots,
)
from attack_analysis.xg_v2 import load_trained_xg_v2_model
from attack_analysis.attack_features import build_attack_features
from attack_analysis.benchmarks import FrozenBenchmarkStore
from attack_analysis.env import (
    load_project_environment,
    openai_api_key_configured,
    openai_model,
    xg_v2_model_path,
)
from attack_analysis.footpass import (
    FootpassSegmentationConfig,
    attack_view_median_x,
    footpass_summary,
    footpass_v2_summary,
    load_footpass_events,
    possession_relative_position,
    prepare_footpass_actions_v2,
    segment_footpass_attacks,
)
from attack_analysis import browser_video as _browser_video
from attack_analysis import dynamic_calibration as _dynamic_calibration
from attack_analysis import pitch_visualization as _pitch_visualization

_browser_video = importlib.reload(_browser_video)
_dynamic_calibration = importlib.reload(_dynamic_calibration)
_pitch_visualization = importlib.reload(_pitch_visualization)
import attack_analysis.attack_v2 as _attack_v2
import attack_analysis.shot_xg as _shot_xg
import attack_analysis.v2_adapters as _v2_adapters
import attack_analysis.xg_v2 as _xg_v2

_xg_v2 = importlib.reload(_xg_v2)
_attack_v2 = importlib.reload(_attack_v2)
_shot_xg = importlib.reload(_shot_xg)
_v2_adapters = importlib.reload(_v2_adapters)
FootpassAttackV2Config = _attack_v2.FootpassAttackV2Config
decode_footpass_attack_v2 = _attack_v2.decode_footpass_attack_v2
load_trained_xg_v2_model = _xg_v2.load_trained_xg_v2_model
score_analysis_shots = _shot_xg.score_analysis_shots
ShotXGRequiredError = _shot_xg.ShotXGRequiredError
ending_event_action = _shot_xg.ending_event_action
is_shot_event = _shot_xg.is_shot_event
sequences_from_video_events = _v2_adapters.sequences_from_video_events
BROWSER_CODECS = _browser_video.BROWSER_CODECS
sample_preview_frames = _browser_video.sample_preview_frames
extract_browser_clip = _browser_video.extract_browser_clip
video_codec = _browser_video.video_codec
plot_freeze_frame = _pitch_visualization.plot_freeze_frame
aligned_event_xy = _pitch_visualization.aligned_event_xy
interactive_pitch_events = _pitch_visualization.interactive_pitch_events
pitch_event_views = _pitch_visualization.pitch_event_views
build_footpass_dynamic_projector = _dynamic_calibration.build_footpass_dynamic_projector
from attack_analysis.event_calibration import load_event_thresholds
from attack_analysis.attack_v2 import FootpassAttackV2Config, decode_footpass_attack_v2
from attack_analysis.attack_v2_shot_insights import (
    GoldOutcomeIndex,
    SEQUENCE_FILTER_ALL_POSSESSIONS,
    SEQUENCE_FILTER_CENSORED_SHOT,
    SEQUENCE_FILTER_CONFIRMED,
    actions_with_resolved_outcome,
    build_shot_insights,
    filter_sequence_entries,
    format_shot_outcome,
    resolve_shot_outcome,
)
from attack_analysis.overview import attack_v2_research_overview
from attack_analysis.prediction_sources import (
    PredictionSourceError,
    load_registry as load_prediction_registry,
    source_choices as prediction_source_choices,
    verify_source as verify_prediction_source,
)
from attack_analysis.gold_annotations import (
    ANNOTATION_LABELS,
    ATTACK_TERMINALS,
    COMPLETENESS_VALUES,
    FORCED_OUT_CAUSES,
    SHOT_OUTCOMES,
    TERMINAL_TYPES,
    GoldSequenceAnnotationV2,
    append_gold_annotation_jsonl,
    append_sealed_test_annotation_jsonl,
    gold_split,
    utc_now_iso,
)
from attack_analysis.llm import (
    DeterministicEvidenceClient,
    OpenAIResponsesClient,
    describe_openai_error,
)
from attack_analysis.models import (
    AttackSequence,
    AttackSequenceV2,
    Event,
    ObservedActionV2,
    PossessionSequenceV2,
)
from attack_analysis.scoreboard_goals import (
    DEFAULT_SCOREBOARD_GOALS,
    SCORING_LOOKBACK_SEC,
    ScoreboardGoalIndex,
    possession_outcomes as scoreboard_possession_outcomes,
)
from attack_analysis.possession import PossessionAssessment, assess_possession
from attack_analysis.research import AttackConclusionV2, EvidenceBundle, EvaluationVerdict, insufficient_evidence_verdict
from attack_analysis.web_search import brave_search_configured, brave_search_context
import attack_analysis.event_inference as _event_inference

_event_inference = importlib.reload(_event_inference)
infer_events_from_observations = _event_inference.infer_events_from_observations
import attack_analysis.footpass_video_events as _footpass_video_events

_footpass_video_events = importlib.reload(_footpass_video_events)
extract_taad_highlights = _footpass_video_events.extract_taad_highlights
import attack_analysis.footpass_dst as _footpass_dst
import attack_analysis.pitch_calibration as _pitch_calibration

_footpass_dst = importlib.reload(_footpass_dst)
_pitch_calibration = importlib.reload(_pitch_calibration)
extract_dst_highlights = _footpass_dst.extract_dst_highlights
KeypointPitchProjector = _pitch_calibration.KeypointPitchProjector
PitchModelConfig = _pitch_calibration.PitchModelConfig
load_or_calibrate_video = _pitch_calibration.load_or_calibrate_video
import attack_analysis.shot_outcome as _shot_outcome

_shot_outcome = importlib.reload(_shot_outcome)
ScoreboardReader = _shot_outcome.ScoreboardReader
attack_with_video_outcome = _shot_outcome.attack_with_video_outcome
infer_attack_outcomes = _shot_outcome.infer_attack_outcomes
from attack_analysis.quality import read_observations_jsonl
from attack_analysis.upload_pipeline import run_cached_video_pipeline, vision_cache_key
from attack_analysis.video import inspect_video
from attack_analysis.video_paths import (
    CALIBRATION_CACHE_KEY,
    OVERLAY_CACHE_KEY,
    PreviewClockProjector,
    VideoPathResolver,
)
from attack_analysis.vision_pipeline import VisionConfig
from attack_analysis.window_analysis import persist_footpass_window_stream


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
load_project_environment(PROJECT_ROOT)
DEFAULT_FOOTPASS = WORKSPACE_ROOT / "FOOTPASS"
DEFAULT_XG_MODEL = PROJECT_ROOT / "models" / "xg-v1.joblib"
DEFAULT_XG_V2_MODEL = xg_v2_model_path(PROJECT_ROOT)
DEFAULT_BENCHMARKS = PROJECT_ROOT / "data" / "benchmarks.jsonl"
DEFAULT_EVENT_PROFILE = PROJECT_ROOT / "models" / "dst-thresholds-v1.json"
DEFAULT_POSE_MODEL = PROJECT_ROOT / "yolo26n-pose.pt"
DEFAULT_DETECTION_MODEL = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_PITCH_MODEL = PROJECT_ROOT / "models" / "football-pitch-detection.pt"
MAX_WINDOW_SEC = 30.0
UPLOAD_EVENT_SOURCES = {
    "footpass_taad": "FOOTPASS TAAD",
    "footpass_dst": "FOOTPASS TAAD → DST (h100, research)",
}
# The clip keeps a little context before the first event and after the finish
# so the shot outcome (save, goal, rebound) stays visible.
UPLOAD_CLIP_LEAD_SEC = 1.0
UPLOAD_CLIP_TAIL_SEC = 3.0
SHOT_OUTCOME_SOURCES = {
    "video_scoreboard": "tỷ số trên scoreboard",
    "video_scoreboard+ball": "tỷ số trên scoreboard + bóng YOLO trong khung thành",
    "video_ball": "bóng YOLO trong khung thành",
}
VIDEO_SHOT_OUTCOME_LABELS = {"goal": "Bàn thắng", "no_goal": "Không thành bàn", "unknown": "Chưa rõ"}
LOGGER = logging.getLogger(__name__)


def show_event_pitch(items, *, heading: str, caption: str, attack_relative: bool = True) -> None:
    positioned = tuple(
        item
        for item in items
        if getattr(item, "x", None) is not None and getattr(item, "y", None) is not None
    )
    if not positioned:
        return
    st.subheader(heading)
    st.caption(
        caption
        + " Di chuột vào một điểm trên sân để xem chi tiết — nhãn không chồng lên bản đồ."
    )
    mirrored = sum(
        view.mirrored_to_same_side
        for view in pitch_event_views(positioned, attack_relative=attack_relative)
    )
    if mirrored:
        st.caption(
            f"{mirrored} điểm cách event liền kề dưới 1.5 giây nhưng lệch quá nửa sân "
            "đã được phản chiếu (x → 105 − x) để nằm cùng mặt sân."
        )
    chart = interactive_pitch_events(positioned, attack_relative=attack_relative)
    st.altair_chart(chart, width="content", theme=None)


def show_yolo_overlay(
    path: Path,
    *,
    heading: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    clip_dir: Path | None = None,
) -> None:
    """Play H.264 overlay in the browser; otherwise show sampled YOLO frames.

    With ``end_sec`` only that window is shown, cut into its own file.
    """

    if not path.is_file():
        return
    st.markdown(f"#### {heading}")
    if end_sec is not None:
        clip = None
        try:
            with st.spinner("Đang cắt video YOLO theo chuỗi sự kiện..."):
                clip = extract_browser_clip(path, start_sec, end_sec, output_dir=clip_dir)
        except Exception:
            LOGGER.exception("Could not cut YOLO overlay clip")
        start_clock, _spoken = _video_clock_parts(start_sec)
        end_clock, _end_spoken = _video_clock_parts(end_sec)
        if clip is not None:
            st.caption(f"Đoạn {start_clock}–{end_clock} của video gốc ({end_sec - start_sec:.1f} giây).")
            st.video(str(clip))
            return
        st.caption(
            f"Không cắt được clip (thiếu FFmpeg). Video phát từ {start_clock} và dừng ở {end_clock}."
        )
        st.video(str(path), start_time=max(0, int(start_sec)), end_time=int(math.ceil(end_sec)))
        return
    codec = ""
    try:
        codec = video_codec(path).strip("\x00")
    except Exception:
        LOGGER.exception("Could not inspect overlay codec")
    if codec in BROWSER_CODECS:
        st.video(str(path), start_time=max(0, int(start_sec)))
        return
    st.warning(
        "OpenCV trên máy này không ghi được H.264 (thiếu OpenH264). "
        "Trình duyệt không phát được file mp4v, nên dưới đây là khung hình overlay YOLO."
    )
    try:
        frames = sample_preview_frames(path, max_frames=8)
    except Exception as error:
        st.error(f"Không đọc được overlay YOLO: {error}")
        return
    if not frames:
        st.error("Overlay YOLO không có khung hình.")
        return
    columns = st.columns(min(4, len(frames)))
    for index, frame in enumerate(frames):
        columns[index % len(columns)].image(frame, channels="BGR", width="stretch")


SEQUENCE_LABELS = {
    "normal_possession": "Possession thông thường",
    "attack": "Đợt tấn công",
    "dangerous_attack": "Đợt tấn công nguy hiểm",
    "insufficient_evidence": "Chưa đủ bằng chứng",
}
SEQUENCE_ICONS = {
    "normal_possession": "⚪",
    "attack": "🟠",
    "dangerous_attack": "🔴",
    "insufficient_evidence": "⚠️",
}


st.set_page_config(page_title="Football Attack Thesis", layout="wide")
st.title("Football Attack-v2 sequence analysis")
st.caption(
    "Possession-first decoding, strict terminal evidence, explicit source tiers and research-only abstention"
)


@st.cache_data(show_spinner=False)
def playbyplay_keys(path: str) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(str(item) for item in payload.get("keys", payload.get("events", {}).keys()))


@st.cache_data(show_spinner="Loading FOOTPASS actions and decoding v2 possessions...")
def load_attack_v2_analysis(
    playbyplay: str,
    match_key: str,
    source: str,
    tactical_h5: str | None,
    max_gap: float,
    min_confidence: float,
    model_name: str | None,
    xg_model_path: str | None,
    xg_model_modified_at: float | None,
    class_threshold_items: tuple[tuple[str, float], ...],
) -> dict[str, object]:
    """Decode one half, joining a same-team rebound inside 3 seconds."""
    del xg_model_modified_at  # Included in the Streamlit cache key.
    tactical_path = Path(tactical_h5) if tactical_h5 else None
    decoder_config = FootpassAttackV2Config(max_observation_gap_sec=max_gap)
    prepared = prepare_footpass_actions_v2(
        Path(playbyplay),
        match_key=match_key,
        source=source,
        tactical_h5=tactical_path,
        min_confidence=min_confidence,
        model_name=model_name,
        class_confidence_thresholds=dict(class_threshold_items),
        decoder_config=decoder_config,
    )
    actions = prepared.actions
    decoded = decode_footpass_attack_v2(actions, decoder_config)

    # Compute the old segmentation only for a clearly labelled ablation count.
    events = load_footpass_events(
        Path(playbyplay),
        match_key=match_key,
        source=source,
        tactical_h5=tactical_path,
        min_confidence=min_confidence,
        model_name=model_name,
        xg_model_path=None,
        class_confidence_thresholds=dict(class_threshold_items),
    )
    attacks = segment_footpass_attacks(
        events,
        match_id=match_key,
        config=FootpassSegmentationConfig(max_gap_sec=max_gap),
    )
    xg_path = Path(xg_model_path) if xg_model_path else None
    loaded_xg = (
        load_trained_xg_v2_model(xg_path, allow_research_only=True)
        if xg_path is not None and xg_path.is_file()
        else None
    )
    xg_payload = score_analysis_shots(
        decoded.attacks,
        decoded.possessions,
        model=loaded_xg,
        tactical_h5=tactical_path,
    )
    return {
        "schema_version": "attack-v2-run/2.0",
        "source_tier": actions[0].source_tier.value if actions else "unknown",
        "observed_actions": [item.to_dict() for item in actions],
        **decoded.to_dict(),
        "summary": footpass_v2_summary(decoded, prepared),
        "excluded_replay_action_ids": list(decoded.excluded_replay_action_ids),
        "xg": xg_payload,
        "legacy_compatibility": {
            "summary": footpass_summary(events, attacks),
            "possession_candidates": [item.to_dict() for item in attacks],
        },
    }


@st.cache_resource(show_spinner=False)
def load_gold_outcome_index():
    return GoldOutcomeIndex.load(PROJECT_ROOT)


SEQUENCE_FILTER_GOALS = "Attack-v2 · bàn thắng"


@st.cache_resource(show_spinner=False)
def load_scoreboard_goal_index(modified: float | None) -> ScoreboardGoalIndex:
    return ScoreboardGoalIndex.load(PROJECT_ROOT / DEFAULT_SCOREBOARD_GOALS)


def scoreboard_goal_index() -> ScoreboardGoalIndex:
    path = PROJECT_ROOT / DEFAULT_SCOREBOARD_GOALS
    return load_scoreboard_goal_index(path.stat().st_mtime if path.is_file() else None)


@st.cache_resource(show_spinner=False)
def load_benchmark_store():
    return FrozenBenchmarkStore.from_jsonl(DEFAULT_BENCHMARKS)


def render_shot_promotion_audit(insights) -> None:
    summary = insights.summary
    audit_columns = st.columns(4)
    audit_columns[0].metric("Live shots", int(summary.get("live_shots", 0)))
    audit_columns[1].metric("Promoted Attack-v2", int(summary.get("promoted_to_attack", 0)))
    audit_columns[2].metric("Shot terminal · censored", int(summary.get("censored_shot_terminal", 0)))
    audit_columns[3].metric(
        "Gold outcomes",
        "Có" if insights.gold_available else "Chưa có",
    )
    if int(summary.get("live_shots", 0)) == 0:
        st.caption("Không có action `shot` live trong hiệp đã chọn.")
        return
    audit_rows = [record.to_dict() for record in insights.records]
    st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)
    if not insights.gold_available:
        st.caption(
            "FOOTPASS oracle không ghi `shot_outcome`. Khi có file "
            "`data/attack-v2-development.jsonl` hoặc `attack-v2-train.jsonl`, UI sẽ gắn "
            "outcome từ gold annotation."
        )


@st.cache_resource(show_spinner=False)
def load_dynamic_projector(
    tactical_h5: str,
    match_key: str,
    start_frame: int,
    end_frame: int,
    video_width: int,
    video_height: int,
    tactical_modified_at: float,
    calibration_policy: int = 4,
):
    del tactical_modified_at  # Included in the Streamlit cache key.
    del calibration_policy
    return build_footpass_dynamic_projector(
        Path(tactical_h5),
        match_key,
        start_frame=start_frame,
        end_frame=end_frame,
        video_width=video_width,
        video_height=video_height,
    )


@st.cache_resource(show_spinner=False)
def load_video_path_resolver() -> VideoPathResolver:
    return VideoPathResolver.from_environment(PROJECT_ROOT)


def match_video_path(match_key: str) -> Path:
    """Low-quality processing video (FOOTPASS extract). Overlay YOLO uses Full HD."""

    return load_video_path_resolver().processing_path(match_key)


def match_preview_video_path(match_key: str) -> Path:
    """Full HD preview video when configured; otherwise falls back to processing video."""

    return load_video_path_resolver().preview_path(match_key)


def possession_to_preview_sequence(
    possession: PossessionSequenceV2,
    *,
    sequence_id: str,
    source: str,
) -> AttackSequence:
    """Legacy sequence used only for video window + conclusion UI."""

    events = tuple(
        Event(
            timestamp=action.timestamp_sec,
            event_type=action.action_type,
            team=possession.possession_team_id,
            x=action.x,
            y=action.y,
            outcome=action.outcome,
            frame=action.frame,
            player_id=action.actor_id,
            jersey=action.jersey,
            role=action.role,
            confidence=action.action_probability if action.action_probability is not None else 1.0,
            source=source,
            evidence_id=action.action_id,
        )
        for action in possession.actions
    )
    if not events:
        raise ValueError("sequence has no actions to preview")
    start_sec = min(possession.start_sec, events[0].timestamp)
    end_sec = max(possession.end_sec, events[-1].timestamp)
    return AttackSequence(
        attack_id=sequence_id,
        team=possession.possession_team_id,
        start_sec=start_sec,
        end_sec=end_sec,
        events=events,
        match_id=possession.match_id,
        start_frame=possession.start_frame,
        end_frame=possession.end_frame,
        source=source,
    )


def build_bundle(sequence: AttackSequence, *, tactical_ready: bool) -> EvidenceBundle:
    features = build_attack_features(sequence)
    assessment = assess_possession(sequence)
    event_ready = len(sequence.events) > 0
    quality_gates = {
        "events": event_ready,
        "tracking": tactical_ready,
        "calibration": tactical_ready and any(item.x is not None for item in sequence.events),
        "attack_qualified": assessment.is_attack,
    }
    limitations = [
        f"Quality gate failed: {name}."
        for name, passed in quality_gates.items()
        if not passed
    ]
    if not assessment.is_attack:
        limitations.append(
            "Sequence is not qualified as an attack; effectiveness evaluation is not applicable."
        )
    return EvidenceBundle(
        attack=features,
        quality_gates=quality_gates,
        quality_metrics={
            "event_confidence": round(sum(item.confidence for item in sequence.events) / len(sequence.events), 4),
            "attack_qualification_score": round(assessment.score / 100.0, 4),
            "attack_position_coverage": assessment.position_coverage,
        },
        source_ids=(sequence.source, assessment.rubric_version),
        limitations=tuple(limitations),
    )


def render_possession_assessment(assessment: PossessionAssessment) -> None:
    label = SEQUENCE_LABELS[assessment.label]
    icon = SEQUENCE_ICONS[assessment.label]
    message = f"{icon} **{label}** · qualification score {assessment.score:.1f}/100"
    if assessment.label == "dangerous_attack":
        st.success(message)
    elif assessment.label == "attack":
        st.warning(message)
    else:
        st.info(message)

    columns = st.columns(4)
    columns[0].metric("Peak progression", f"{assessment.peak_progression_m:.1f} m")
    columns[1].metric(
        "Vị trí xa nhất",
        "—" if assessment.furthest_x is None else f"x={assessment.furthest_x:.1f} m",
    )
    columns[2].metric("Box touches", assessment.penalty_box_touches)
    columns[3].metric("Position evidence", f"{100 * assessment.position_coverage:.0f}%")
    st.markdown("  \n".join(f"- {reason}" for reason in assessment.reasons))
    if assessment.limitations:
        st.caption("Giới hạn: " + " ".join(assessment.limitations))
    st.caption(
        "Đây là rubric baseline để tách attack khỏi possession, không phải xác suất ML. "
        f"Phiên bản: {assessment.rubric_version}."
    )


def render_attack(sequence: AttackSequence, bundle: EvidenceBundle, video_path: Path | None) -> None:
    assessment = assess_possession(sequence)
    st.subheader("Phân loại chuỗi possession")
    render_possession_assessment(assessment)

    left, middle, right = st.columns(3)
    left.metric("Events", bundle.attack.event_count)
    middle.metric("Progression", f"{bundle.attack.progression_m:.1f} m")
    right.metric("xG", f"{bundle.attack.xg_sum:.3f}")

    xg_sources = sorted(
        {item.xg_source for item in sequence.events if item.xg is not None and item.xg_source}
    )
    if xg_sources:
        trained = any(source.startswith("trained_xg:") for source in xg_sources)
        st.caption(
            ("xG đang dùng model đã huấn luyện · " if trained else "xG đang dùng baseline · ")
            + ", ".join(xg_sources)
        )

    gate_labels = {
        "events": "Events",
        "tracking": "Tracking",
        "calibration": "Calibration",
        "attack_qualified": "Attack qualified",
    }
    gate_columns = st.columns(len(bundle.quality_gates))
    for column, (name, passed) in zip(gate_columns, bundle.quality_gates.items()):
        column.metric(gate_labels.get(name, name.replace("_", " ").title()), "PASS" if passed else "FAIL")

    if video_path and video_path.exists():
        st.video(str(video_path), start_time=max(0, int(sequence.start_sec)))

    event_rows = [item.to_dict() for item in sequence.events]
    st.subheader("Event timeline")
    st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)
    show_event_pitch(
        sequence.events,
        heading="Bản đồ đợt tấn công" if assessment.is_attack else "Bản đồ chuỗi possession",
        caption=(
            "Sân 105 × 68 m (tỉ lệ bóng đá, 1 m ngang = 1 m dọc), hướng tấn công trái → phải. "
            "Hover từng điểm để đọc đầy đủ nội dung event."
        ),
    )

    with st.expander("Dữ liệu đầu vào có cấu trúc (JSON, dành cho nghiên cứu)"):
        st.json(bundle.to_dict(), expanded=False)


def _video_clock_parts(seconds: float) -> tuple[str, str]:
    """Return player clock ``m:ss.s`` and a Vietnamese minute/second reading."""

    safe = max(0.0, float(seconds))
    minutes = int(safe // 60)
    remainder = safe - minutes * 60
    clock = f"{minutes}:{remainder:04.1f}"
    if minutes == 0:
        spoken = f"{remainder:.1f} giây"
    else:
        spoken = f"{minutes} phút {remainder:.1f} giây"
    return clock, spoken


def _video_clock_label(seconds: float) -> str:
    clock, spoken = _video_clock_parts(seconds)
    return f"{clock} ({spoken})"


def render_calibrated_window(
    sequence: AttackSequence,
    *,
    match_key: str,
    video_path: Path,
    playbyplay_path: Path,
    tactical_path: Path,
    video_resolver: VideoPathResolver | None = None,
) -> None:
    st.subheader("Phân tích video theo khung thời gian")
    resolver = video_resolver or load_video_path_resolver()
    overlay_path = resolver.yolo_video_path(match_key)
    if not overlay_path.is_file():
        overlay_path = video_path
    processing_path = resolver.processing_path(match_key)
    try:
        using_fullhd = overlay_path.is_file() and (
            not processing_path.is_file()
            or overlay_path.resolve() != processing_path.resolve()
        )
    except OSError:
        using_fullhd = overlay_path.is_file() and str(overlay_path) != str(processing_path)
    st.caption(
        "Chọn một cửa sổ ngắn quanh chuỗi possession. YOLO chỉ chạy trong cửa sổ này; "
        "frame/timestamp FOOTPASS vẫn giữ theo toàn trận để khớp event và tactical HDF5."
        + (
            f" Overlay YOLO xuất từ Full HD `{overlay_path.name}` "
            f"({resolver.preview_path(match_key).parent})."
            if using_fullhd and overlay_path.is_file()
            else ""
        )
    )
    if not overlay_path.is_file() or not playbyplay_path.is_file() or not tactical_path.is_file():
        missing = []
        if not overlay_path.is_file():
            missing.append(str(overlay_path))
        if not playbyplay_path.is_file():
            missing.append(str(playbyplay_path))
        if not tactical_path.is_file():
            missing.append(str(tactical_path))
        st.warning("Thiếu dữ liệu để chạy calibrated window: " + ", ".join(missing))
        return

    metadata = inspect_video(overlay_path)
    widget_key = hashlib.sha1(
        f"{sequence.attack_id}|{sequence.source}|fullhd-overlay".encode("utf-8")
    ).hexdigest()[:12]
    preview_default_start = resolver.match_sec_to_preview_sec(match_key, max(0.0, sequence.start_sec - 1.0))
    preview_default_end = resolver.match_sec_to_preview_sec(
        match_key,
        max(sequence.end_sec + 1.0, sequence.start_sec + 1.0),
    )
    default_start = max(0.0, preview_default_start)
    default_end = min(metadata.duration_sec, max(preview_default_end, default_start + 2.0))
    st.caption("Ô nhập vẫn là giây. Dòng dưới mỗi ô là mốc phút:giây trên thanh phát video.")
    start_column, end_column = st.columns(2)
    start_sec = start_column.number_input(
        "Bắt đầu (giây video overlay)",
        min_value=0.0,
        max_value=float(metadata.duration_sec),
        value=float(round(default_start, 2)),
        step=0.5,
        key=f"window_start_{widget_key}",
    )
    start_column.caption(f"Trên video: {_video_clock_label(float(start_sec))}")
    end_sec = end_column.number_input(
        "Kết thúc (giây video overlay)",
        min_value=0.0,
        max_value=float(metadata.duration_sec),
        value=float(round(default_end, 2)),
        step=0.5,
        key=f"window_end_{widget_key}",
    )
    end_column.caption(f"Trên video: {_video_clock_label(float(end_sec))}")
    selected_duration = end_sec - start_sec
    start_sec = max(float(start_sec), float(end_sec) - MAX_WINDOW_SEC)
    duration = end_sec - start_sec
    start_clock, _start_spoken = _video_clock_parts(float(start_sec))
    end_clock, _end_spoken = _video_clock_parts(float(end_sec))
    st.caption(
        f"Cửa sổ {start_clock} → {end_clock} · dài {duration:.1f} giây · "
        f"khoảng {max(0, round(duration * metadata.fps)):,} frame Full HD. "
        f"Tối đa {MAX_WINDOW_SEC:.0f} giây để phù hợp GPU 6 GB và giữ thời gian demo hợp lý."
    )
    if selected_duration > MAX_WINDOW_SEC:
        st.info(
            f"Cửa sổ đã chọn dài {selected_duration:.1f} giây. Chỉ phân tích "
            f"{MAX_WINDOW_SEC:.0f} giây cuối ({start_clock} → {end_clock}), "
            "là đoạn dẫn tới cú dứt điểm."
        )

    result_key = f"calibrated_window_result_{widget_key}_{start_sec:.2f}_{end_sec:.2f}"
    if st.button(
        "Chạy FOOTPASS-assisted calibration + YOLO cho cửa sổ này",
        type="primary",
        key=f"run_window_{widget_key}",
    ):
        if duration <= 0:
            st.error("Thời điểm kết thúc phải lớn hơn thời điểm bắt đầu.")
        else:
            try:
                start_frame = max(0, int(round(start_sec * metadata.fps)))
                end_frame = min(
                    metadata.frame_count - 1,
                    int(round(end_sec * metadata.fps)),
                )
                match_start_frame = resolver.preview_frame_to_match_frame(
                    match_key,
                    start_frame,
                    metadata.fps,
                )
                match_end_frame = resolver.preview_frame_to_match_frame(
                    match_key,
                    end_frame,
                    metadata.fps,
                )
                with st.spinner("Đang fit homography động và kiểm tra holdout error..."):
                    projector = load_dynamic_projector(
                        str(tactical_path),
                        match_key,
                        match_start_frame,
                        match_end_frame,
                        metadata.width,
                        metadata.height,
                        tactical_path.stat().st_mtime,
                    )
                    frame_projector = PreviewClockProjector(
                        inner=projector,
                        match_key=match_key,
                        resolver=resolver,
                        preview_fps=metadata.fps,
                    )
                report = projector.report.to_dict()
                try:
                    import torch

                    device = "0" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
                stat = overlay_path.stat()
                video_identity = hashlib.sha256(
                    f"{overlay_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
                ).hexdigest()
                overlay_size = 1280 if metadata.width >= 1280 else 960
                vision_config = VisionConfig(
                    pose_model=str(DEFAULT_POSE_MODEL),
                    detection_model=str(DEFAULT_DETECTION_MODEL),
                    device=device,
                    image_size=overlay_size,
                    start_frame=start_frame,
                    max_frames=end_frame - start_frame + 1,
                    field_lines=True,
                    player_confidence=0.15,
                    keypoint_confidence=0.2,
                )
                if int(report.get("valid_sampled_frames") or 0) == 0:
                    st.warning(
                        "Oracle homography FAIL trên cửa sổ này — không có nghĩa cả pipeline hỏng. "
                        "Event FOOTPASS vẫn đúng; freeze-frame tactical HDF5 (nếu có) vẫn dùng cho xG. "
                        "YOLO vẫn chạy (bbox/skeleton). Tọa độ sân YOLO sẽ lấy grass-polygon "
                        "video-only nếu detect được đường biên cỏ — đây là baseline nghiên cứu, không phải oracle."
                    )
                    yolo_projector = None
                else:
                    yolo_projector = frame_projector
                video_clock = resolver.video_clock(match_key, preview_fps=metadata.fps)
                sync_mtime = (
                    resolver.sync_path.stat().st_mtime
                    if resolver.sync_path is not None and resolver.sync_path.is_file()
                    else None
                )
                with st.spinner(
                    "Đang chạy YOLO trên Full HD, chiếu track lên mặt sân và áp dụng event quality gates..."
                ):
                    window_manifest = run_cached_video_pipeline(
                        overlay_path,
                        PROJECT_ROOT / "runs" / "window-cache",
                        video_identity,
                        vision_config,
                        projector=yolo_projector,
                        xg_model_path=(DEFAULT_XG_MODEL if DEFAULT_XG_MODEL.is_file() else None),
                        source_tier="predicted_actions_oracle_tactical",
                        dataset_paths=(playbyplay_path, tactical_path),
                        cache_context={
                            "calibration": CALIBRATION_CACHE_KEY,
                            "overlay": OVERLAY_CACHE_KEY,
                            "match_key": match_key,
                            "tactical_mtime": tactical_path.stat().st_mtime,
                            "video_sync_mtime": sync_mtime,
                            "preview_offset_sec": video_clock.get("preview_offset_sec"),
                            "xg_model_mtime": (
                                DEFAULT_XG_MODEL.stat().st_mtime
                                if DEFAULT_XG_MODEL.is_file()
                                else None
                            ),
                        },
                    )
                    window_manifest = dict(window_manifest)
                    window_manifest["video_clock"] = resolver.video_clock(
                        match_key,
                        preview_fps=metadata.fps,
                    )
                    window_manifest["calibration"] = report
                    try:
                        st.session_state[result_key] = persist_footpass_window_stream(
                            window_manifest,
                            sequence,
                            start_sec=resolver.preview_sec_to_match_sec(
                                match_key, start_sec
                            ),
                            end_sec=resolver.preview_sec_to_match_sec(
                                match_key, end_sec
                            ),
                        )
                    except Exception as persist_error:
                        LOGGER.exception("Window stream persist failed after YOLO")
                        window_manifest = dict(window_manifest)
                        window_manifest["persist_warning"] = str(persist_error)
                        st.session_state[result_key] = window_manifest
                        st.warning(
                            "YOLO đã chạy xong nhưng không lưu được PossessionCandidateV1: "
                            f"{persist_error}"
                        )
            except Exception as error:
                LOGGER.exception("Calibrated FOOTPASS window failed")
                st.error(f"Không thể phân tích cửa sổ đã chọn: {error}")

    manifest = st.session_state.get(result_key)
    if manifest:
        st.session_state[f"window_manifest_{sequence.attack_id}"] = manifest
    if not manifest:
        st.info(
            "FOOTPASS-assisted calibration dùng bottom-center ROI của cầu thủ làm correspondence "
            "với tactical coordinates và đánh giá trên 25% điểm holdout ở từng frame. Đây là đường "
            "oracle calibration; artifact luôn mang tier `predicted_actions_oracle_tactical` và hash "
            "cả play-by-play lẫn tactical HDF5. CV-only vẫn có gate riêng."
        )
        return

    calibration = dict(manifest.get("calibration") or {})
    calibration_passed = calibration.get("status") == "pass"
    calibration_columns = st.columns(4)
    calibration_columns[0].metric(
        "Calibration", "PASS" if calibration_passed else "FAIL"
    )
    calibration_columns[1].metric(
        "Eligible coverage", f"{100 * float(calibration.get('projected_frame_coverage', 0)):.1f}%"
    )
    median_error = calibration.get("median_holdout_error_m")
    p90_error = calibration.get("p90_holdout_error_m")
    calibration_columns[2].metric(
        "Median holdout error", "—" if median_error is None else f"{median_error:.2f} m"
    )
    calibration_columns[3].metric(
        "P90 holdout error", "—" if p90_error is None else f"{p90_error:.2f} m"
    )
    direct_coverage = 100 * float(calibration.get("direct_frame_coverage", 0))
    interpolated_coverage = 100 * float(
        calibration.get("interpolated_frame_coverage", 0)
    )
    extrapolated_coverage = 100 * float(
        calibration.get("extrapolated_frame_coverage", 0)
    )
    unsupported_frames = int(calibration.get("unsupported_frames", 0))
    ineligible_frames = int(calibration.get("ineligible_frames", 0))
    eligible_frames = int(calibration.get("eligible_frames") or 0)
    video_clock = dict(manifest.get("video_clock") or {})
    if video_clock:
        st.caption(
            f"Đồng hồ video: `{video_clock.get('role')}` · preview {video_clock.get('preview_fps')} fps · "
            f"trận {video_clock.get('match_fps')} fps · offset {video_clock.get('preview_offset_sec')} s. "
            "Sai offset/fps làm cửa sổ HDF5 trống (coverage 0%) dù event FOOTPASS đúng. "
            f"Cache overlay `{OVERLAY_CACHE_KEY}` — chạy lại cửa sổ sau khi đổi sync JSON."
        )
    if any((direct_coverage, interpolated_coverage, extrapolated_coverage, ineligible_frames)):
        st.caption(
            f"Nguồn calibration: {direct_coverage:.1f}% fit trực tiếp qua holdout gate · "
            f"{interpolated_coverage:.1f}% nội suy giữa hai frame hợp lệ · "
            f"{extrapolated_coverage:.1f}% ngoại suy giới hạn · "
            f"{unsupported_frames} frame live không chiếu được · "
            f"{ineligible_frames} frame replay/không bbox (loại khỏi quality gate"
            + (f", còn {eligible_frames} frame đủ điều kiện" if eligible_frames else "")
            + ")."
        )

    outputs = dict(manifest.get("outputs") or {})
    annotated_path = Path(str(outputs.get("annotated_video") or ""))
    show_yolo_overlay(annotated_path, heading="Video YOLO đã xử lý trên sự kiện")

    quality = dict(manifest.get("quality") or {})
    if quality:
        quality_columns = st.columns(4)
        quality_columns[0].metric(
            "Players/frame", f"{float(quality.get('average_players_per_frame', 0)):.1f}"
        )
        quality_columns[1].metric(
            "Ball coverage", f"{100 * float(quality.get('frames_with_ball_ratio', 0)):.1f}%"
        )
        quality_columns[2].metric(
            "Player tracking", f"{100 * float(quality.get('player_tracking_coverage', 0)):.1f}%"
        )
        quality_columns[3].metric(
            "Calibrated objects", f"{100 * float(quality.get('calibrated_position_coverage', 0)):.1f}%"
        )

    window_events = tuple(
        event
        for event in sequence.events
        if start_sec <= event.timestamp <= end_sec
    )
    stream_summary = dict(manifest.get("structured_event_stream") or {})
    legacy_gates = dict(stream_summary.get("quality_gates") or {})
    skip_reason = str(manifest.get("analysis_skip_reason") or "")
    st.markdown("#### Overlay YOLO trên cửa sổ đã chọn")
    st.caption(
        "Attack-v2 đã được decode ở trên. Overlay này chiếu track YOLO bằng homography FOOTPASS "
        "(hoặc grass-polygon video-only khi oracle FAIL) và không đổi hợp đồng v2, không gắn xG, "
        "không đưa effectiveness verdict."
    )
    if skip_reason == "dynamic_calibration_quality_gate_failed":
        st.warning(
            "Oracle calibration quality gate FAIL — overlay YOLO vẫn được xuất. "
            "Không suy event từ video. xG dùng FOOTPASS (+ freeze HDF5 nếu có). "
            "Tọa độ sân YOLO có thể là video-only grass polygon (baseline nghiên cứu)."
        )
    elif skip_reason == "calibration_required_for_event_inference":
        st.info(
            "Không có homography oracle trên cửa sổ này. YOLO overlay vẫn chạy; "
            "nếu thấy polygon cỏ, fusion có thể dùng video-only baseline."
        )
    elif not calibration_passed:
        st.info(
            "Quality gate coverage trên frame đủ điều kiện chưa đạt 90%. "
            "YOLO vẫn chạy trên các frame đã chiếu được; replay/thiếu bbox không tính vào gate. "
            "Event FOOTPASS không bị ảnh hưởng."
        )
    gate_columns = st.columns(4)
    gate_columns[0].metric("Assisted calibration", "PASS" if calibration_passed else "FAIL")
    gate_columns[1].metric(
        "Window boundary",
        str(stream_summary.get("boundary_status") or "unknown").upper(),
    )
    gate_columns[2].metric(
        "Terminal evidence",
        "PASS" if legacy_gates.get("terminal_evidence_complete") else "INCOMPLETE",
    )
    cv_gates = dict(manifest.get("event_quality_gates") or {})
    cv_event_passed = bool(cv_gates) and all(cv_gates.values())
    gate_columns[3].metric("CV event gate", "PASS" if cv_event_passed else "ABSTAIN")
    st.caption(
        "FOOTPASS action/tactical và YOLO là hai nguồn song song. Chỉ source tier `fused` "
        "với association và field-level provenance mới được phép kết hợp hai nguồn."
    )

    if window_events:
        timeline_rows = []
        aligned_window = aligned_event_xy(window_events)
        for event, (pitch_x, pitch_y, mirrored) in zip(window_events, aligned_window, strict=True):
            if pitch_x is None or pitch_y is None:
                pitch_label = "—"
            elif mirrored:
                pitch_label = f"({pitch_x:.1f}, {pitch_y:.1f}) · cùng mặt sân"
            else:
                pitch_label = f"({pitch_x:.1f}, {pitch_y:.1f})"
            timeline_rows.append(
                {
                    "Thời gian": f"{event.timestamp:.2f}s",
                    "Sự kiện": ACTION_LABELS.get(event.event_type, event.event_type.title()),
                    "Đội": event.team,
                    "Cầu thủ": f"#{event.jersey}" if event.jersey is not None else "—",
                    "Tọa độ sân": pitch_label,
                    "Tin cậy": f"{100 * event.confidence:.1f}%",
                    "Legacy xG evidence": (
                        "—" if event.xg is None else "Có (không đánh giá)"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(timeline_rows), width="stretch", hide_index=True)
        show_event_pitch(
            window_events,
            heading="Bản đồ event trong cửa sổ",
            caption=(
                "Mỗi điểm trên sân tương ứng một event trong cửa sổ. "
                "Hover để xem thời điểm, đội, cầu thủ, tọa độ, kết quả và độ tin cậy."
            ),
        )
    else:
        st.warning("Không có event của possession candidate đã chọn trong cửa sổ này.")

    legacy_xg_count = sum(event.xg is not None for event in window_events)
    if legacy_xg_count:
        st.caption(
            f"Có {legacy_xg_count} trường xG cũ trong evidence thô. Chúng không được render "
            "thành xác suất hay dùng cho verdict; xG-v2 chỉ chạy sau terminal v2 đã xác nhận."
        )

    with st.expander("Sidecar V1 (chỉ để audit, không phải kết luận)"):
        st.caption(
            "Artifact cửa sổ vẫn được lưu như PossessionCandidateV1/`legacy_unvalidated` "
            "để không trộn overlay YOLO vào AttackSequenceV2."
        )
        st.json(
            {
                "calibration": calibration,
                "event_quality_gates": cv_gates,
                "analysis_status": manifest.get("analysis_status"),
                "analysis_skip_reason": manifest.get("analysis_skip_reason"),
                "outputs": outputs,
            },
            expanded=False,
        )


VERDICT_LABELS = {
    "effective": ("Hiệu quả", "✅"),
    "promising": ("Có triển vọng", "🔎"),
    "ineffective": ("Không hiệu quả", "⚠️"),
    "insufficient_evidence": ("Chưa đủ bằng chứng", "⛔"),
}
ACTION_LABELS = {
    "drive": "Dẫn bóng",
    "carry": "Dẫn bóng",
    "pass": "Chuyền bóng",
    "possession_start": "Bắt đầu kiểm soát",
    "turnover": "Mất bóng",
    "shot_candidate": "Dứt điểm (ước lượng)",
    "cross": "Tạt bóng",
    "header": "Đánh đầu",
    "headed_attempt": "Đánh đầu về cầu môn",
    "headed_shot": "Đánh đầu về cầu môn",
    "shot": "Dứt điểm",
    "shot_attempt": "Dứt điểm",
    "block": "Chặn bóng",
    "tackle": "Tắc bóng",
    "throw_in": "Ném biên",
    "goal": "Bàn thắng",
}
SOURCE_LABELS = {
    "footpass_oracle": "FOOTPASS · Oracle",
    "footpass_predicted": "FOOTPASS · Predicted",
}


def render_conclusion_v2(
    conclusion: AttackConclusionV2,
    evidence,
    overview: dict[str, object] | None,
) -> None:
    st.markdown("### Kết luận Attack-v2 (research-only)")
    st.caption("Không gán nhãn hiệu quả categorical. xG-v2 là research_only cho tới human-domain validation.")
    if overview:
        summary_cols = st.columns(4)
        xg_value = overview.get("xg")
        summary_cols[0].metric("Terminal", str(overview.get("terminal_type") or "—"))
        summary_cols[1].metric(
            "xG-v2",
            "—" if xg_value is None else f"{float(xg_value):.3f}",
        )
        summary_cols[2].metric("Chance quality", conclusion.chance_quality)
        summary_cols[3].metric("Actions", int(overview.get("action_count") or 0))
        if overview.get("xg_status") != "scored":
            st.info(f"xG status: `{overview.get('xg_status')}` · {evidence.withheld_reason or 'chưa fusion'}")
        else:
            st.caption(f"xG model `{overview.get('xg_model_id')}` · `{evidence.xg_release_status}`")

    st.markdown("#### 1. Chance quality")
    if conclusion.xg_probability is not None:
        st.write(f"xG-v2 = **{conclusion.xg_probability:.3f}** · band `{conclusion.chance_quality}`")
    else:
        st.write("Chưa có xác suất xG. Có thể đọc chiến thuật mà không gắn số cơ hội.")

    st.markdown("#### 2. Spatial context")
    fusion = evidence.fusion
    if fusion is None:
        st.caption("Chưa có YOLO overlay hay freeze-frame FOOTPASS cho sequence này.")
    else:
        assoc = fusion.association
        xg_raw = dict(assoc.xg_raw or {})
        spatial = dict(assoc.spatial_context or {})
        provenance = dict(spatial.get("provenance") or {})
        cols = st.columns(5)
        gk_distance = spatial.get("gk_distance_m")
        cols[0].metric("Association", assoc.method)
        cols[1].metric("Track", assoc.track_id or "—")
        cols[2].metric(
            "GK distance",
            "—" if gk_distance is None else f"{float(gk_distance):.1f} m",
        )
        cols[3].metric("Foes", xg_raw.get("foe_players") if xg_raw.get("foe_players") is not None else "—")
        cols[4].metric("Body part", str(xg_raw.get("body_part") or "—"))
        buildup = dict(spatial.get("possession_buildup") or {})
        assist = dict(buildup.get("assist") or {})
        if assist:
            st.caption(
                f"Build-up: `{assist.get('action_type')}` "
                f"{float(assist['dt_sec']):.1f}s trước terminal"
                + (" · cross → header" if buildup.get("cross_then_header") else "")
            )
        rows = []
        display_keys = (
            ("x_shooter", "Shooter x"),
            ("y_shooter", "Shooter y"),
            ("body_part", "Body part"),
            ("x_goalkeeper", "GK x"),
            ("y_goalkeeper", "GK y"),
            ("goalkeeper_visible", "GK (oracle)"),
            ("foe_players", "Foe players"),
            ("ally_players", "Allies (ex. shooter)"),
            ("defending_players", "Defenders ≤2 m"),
            ("under_pressure", "Under pressure"),
            ("play_pattern", "Play pattern"),
            ("shot_type", "Shot type"),
            ("shot_angle", "Shot angle (rad)"),
            ("distance_to_goal", "Distance to goal (m)"),
        )
        for key, label in display_keys:
            value = xg_raw.get(key)
            if key in {"defenders_in_box", "allies_ahead", "foes_in_shot_corridor", "gk_distance_m"}:
                value = spatial.get(key, value)
            rows.append(
                {
                    "Field": label,
                    "Value": value if value is not None else "—",
                    "Source": provenance.get(key, "—"),
                }
            )
        for key, label in (
            ("defenders_in_box", "Defenders in box"),
            ("allies_ahead", "Allies ahead of ball"),
            ("foes_in_shot_corridor", "Foes on shot line"),
            ("goalkeeper_in_camera", "GK in camera (YOLO)"),
            ("yolo_player_count", "YOLO players with pitch xy"),
            ("ball_detected", "Ball in YOLO frame"),
            ("shooter_keypoint_count", "Shooter keypoints"),
            ("calibration_confidence", "Frame calibration"),
        ):
            value = spatial.get(key)
            rows.append(
                {
                    "Field": label,
                    "Value": value if value is not None else "—",
                    "Source": provenance.get(
                        key,
                        "yolo"
                        if key.startswith("yolo")
                        or key in {"ball_detected", "goalkeeper_in_camera", "shooter_keypoint_count", "calibration_confidence"}
                        else "—",
                    ),
                }
            )
        if buildup.get("assist_type"):
            rows.append(
                {
                    "Field": "Last assist",
                    "Value": f"{buildup['assist_type']} ({assist.get('dt_sec')}s)",
                    "Source": "event_sequence",
                }
            )
        if spatial.get("play_pattern_reason"):
            rows.append(
                {
                    "Field": "Play pattern note",
                    "Value": spatial.get("play_pattern_reason"),
                    "Source": provenance.get("play_pattern", spatial.get("play_pattern_source") or "event_sequence"),
                }
            )
        align = spatial.get("yolo_role_align") if isinstance(spatial.get("yolo_role_align"), dict) else {}
        if align:
            rows.append(
                {
                    "Field": "YOLO↔FOOTPASS role match",
                    "Value": f"{align.get('n_matched')} tracks · agreement {align.get('agreement')}",
                    "Source": provenance.get("yolo_role_align", "yolo+footpass_tactical"),
                }
            )
        if spatial.get("ball_quality_warning"):
            rows.append(
                {
                    "Field": "Ball quality",
                    "Value": spatial["ball_quality_warning"],
                    "Source": "yolo",
                }
            )
        st.dataframe(pd.DataFrame(rows).astype({"Value": str}), width="stretch", hide_index=True)
        if spatial.get("ball_quality_warning"):
            st.caption(
                f"{spatial['ball_quality_warning']} — xG vẫn được tính từ vị trí sút "
                "(missingness masks cho GK/counts), không withhold cả model."
            )
        if spatial.get("ontology_note"):
            st.caption(str(spatial["ontology_note"]))
        players = list(spatial.get("players") or ())
        if players or buildup.get("events"):
            freeze_figure = plot_freeze_frame(
                players,
                x_shooter=xg_raw.get("x_shooter"),
                y_shooter=xg_raw.get("y_shooter"),
                x_goalkeeper=xg_raw.get("x_goalkeeper"),
                y_goalkeeper=xg_raw.get("y_goalkeeper"),
                events=tuple(buildup.get("events") or ()),
                yolo_shooter=spatial.get("yolo_shooter") if isinstance(spatial.get("yolo_shooter"), dict) else None,
                shooter_keypoints=tuple(spatial.get("shooter_keypoints") or ()),
                shooter_bbox=spatial.get("shooter_bbox") if isinstance(spatial.get("shooter_bbox"), dict) else None,
            )
            st.pyplot(freeze_figure, width="stretch")
            plt.close(freeze_figure)
        with st.expander("xg_raw JSON"):
            st.json(xg_raw, expanded=False)

    st.markdown("#### 3. So sánh benchmark")
    if evidence.benchmarks:
        st.dataframe(
            pd.DataFrame([item.to_dict() for item in evidence.benchmarks]),
            width="stretch",
            hide_index=True,
        )
        if conclusion.comparison_to_role_benchmark:
            st.write(conclusion.comparison_to_role_benchmark)
    else:
        st.caption("Không có frozen benchmark khớp. Chạy `build-footpass-benchmarks` hoặc dùng `data/benchmarks.jsonl`.")

    st.markdown("#### 4. Bối cảnh web (không dùng làm số liệu)")
    if evidence.web_context:
        for snippet in evidence.web_context:
            title = str(snippet.get("title") or "Kết quả")
            url = str(snippet.get("url") or "")
            description = str(snippet.get("description") or "")
            st.markdown(f"- [{title}]({url}) — {description}" if url else f"- {title} — {description}")
        st.caption("Brave Search: context only, không cite cho xG/benchmark.")
    else:
        st.caption("Không có snippet web. Đặt `BRAVE_SEARCH_API_KEY` để bật.")

    st.markdown("#### 5. Tactical read")
    st.write(conclusion.tactical_read)
    if conclusion.structural_features:
        for item in conclusion.structural_features:
            st.write(f"- {item}")
    for claim in conclusion.claims:
        st.write(f"- {claim.text} (`{'`, `'.join(claim.evidence_ids)}`)")
    if conclusion.limitations:
        with st.expander("Limitations"):
            for item in conclusion.limitations:
                st.write(f"• {item}")
    with st.expander("Provenance JSON"):
        st.json({"overview": overview, "conclusion": conclusion.to_dict(), "evidence": evidence.to_dict()}, expanded=False)


def render_verdict(
    verdict: EvaluationVerdict,
    bundle: EvidenceBundle,
    sequence: AttackSequence,
) -> None:
    label, icon = VERDICT_LABELS[verdict.label]
    model_name = verdict.model or "không xác định"
    confidence_percent = round(verdict.confidence * 100)

    st.markdown(f"### {icon} Kết luận: {label}")
    score_column, confidence_column, model_column = st.columns(3)
    score_column.metric("Điểm hiệu quả", f"{verdict.score}/5")
    confidence_column.metric("Độ tin cậy", f"{confidence_percent}%")
    model_column.metric("Mô hình", model_name)

    if verdict.abstained:
        st.error(
            "Hệ thống chưa thể đưa ra kết luận đáng tin cậy vì dữ liệu chưa vượt qua các quality gate bắt buộc."
        )
    else:
        st.info(
            f"Hệ thống đánh giá đợt tấn công **{label.lower()}**, đạt **{verdict.score}/5 điểm** "
            f"với độ tin cậy **{confidence_percent}%**. Các nhận định dưới đây chỉ sử dụng dữ liệu "
            "có cấu trúc và đều có thể truy ngược về bằng chứng."
        )

    cited_ids: list[str] = []
    for claim in verdict.claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id not in cited_ids:
                cited_ids.append(evidence_id)
    citation_numbers = {evidence_id: index + 1 for index, evidence_id in enumerate(cited_ids)}

    if verdict.claims:
        st.markdown("#### Vì sao hệ thống đưa ra kết luận này?")
        for claim in verdict.claims:
            links = " ".join(
                (
                    f'<a href="#evidence-{citation_numbers[evidence_id]}" '
                    f'title="Xem bằng chứng {citation_numbers[evidence_id]}">'
                    f'[{citation_numbers[evidence_id]}]</a>'
                )
                for evidence_id in claim.evidence_ids
            )
            st.markdown(
                f"- {html.escape(claim.text)} {links}",
                unsafe_allow_html=True,
            )

    if verdict.limitations:
        with st.expander(f"Giới hạn của kết luận ({len(verdict.limitations)})"):
            for limitation in verdict.limitations:
                st.write(f"• {limitation}")

    event_by_evidence = {
        event.evidence_id: event for event in sequence.events if event.evidence_id is not None
    }
    benchmark_by_evidence = {record.benchmark_id: record for record in bundle.benchmarks}
    with st.expander(f"Nguồn và bằng chứng được trích dẫn ({len(cited_ids)})", expanded=True):
        st.caption(
            "Bấm vào số nguồn trong phần nhận định để đi tới bản ghi tương ứng. "
            "Mỗi bản ghi giữ nguyên evidence ID để có thể kiểm toán lại."
        )
        for evidence_id in cited_ids:
            reference = citation_numbers[evidence_id]
            st.markdown(f'<div id="evidence-{reference}"></div>', unsafe_allow_html=True)
            event = event_by_evidence.get(evidence_id)
            benchmark = benchmark_by_evidence.get(evidence_id)
            with st.container(border=True):
                if event is not None:
                    action_label = ACTION_LABELS.get(event.event_type, event.event_type.title())
                    st.markdown(
                        f"**[{reference}] {action_label} · {event.timestamp:.2f}s**"
                    )
                    player = (
                        f"áo số {event.jersey}"
                        if event.jersey is not None
                        else (f"player {event.player_id}" if event.player_id else "chưa xác định")
                    )
                    location = (
                        f"({event.x:.1f}, {event.y:.1f})"
                        if event.x is not None and event.y is not None
                        else "không có"
                    )
                    st.write(
                        f"Đội {event.team} · {player} · vai trò {event.role or 'chưa xác định'} · "
                        f"tọa độ {location} · độ tin cậy {event.confidence:.2f}"
                    )
                    source_label = SOURCE_LABELS.get(event.source, event.source)
                    st.caption(f"Nguồn: {source_label} · Evidence ID: {evidence_id}")
                elif benchmark is not None:
                    subject = benchmark.player_name or benchmark.role
                    st.markdown(f"**[{reference}] Benchmark: {subject}**")
                    st.write(
                        f"{benchmark.metric}: {benchmark.value_per90:.2f}/90 · "
                        f"percentile {benchmark.percentile:.1f} · mùa {benchmark.season}"
                    )
                    st.markdown(
                        f"[Mở nguồn benchmark]({benchmark.source_url}) · snapshot {benchmark.snapshot_date}"
                    )
                    st.caption(f"Benchmark ID: {evidence_id}")
                else:
                    st.markdown(f"**[{reference}] Nguồn dữ liệu: {evidence_id}**")
                    st.caption("Bản ghi nguồn thuộc EvidenceBundle đã đóng băng của đợt tấn công.")

    with st.expander("Dữ liệu nền và provenance (JSON dành cho nghiên cứu/debug)"):
        evidence_tab, verdict_tab = st.tabs(["EvidenceBundle", "EvaluationVerdict"])
        with evidence_tab:
            st.json(bundle.to_dict(), expanded=False)
        with verdict_tab:
            st.json(verdict.to_dict(), expanded=False)


def _prediction_file_ok(registry: object, source: object) -> bool:
    try:
        verify_prediction_source(registry, source)  # type: ignore[arg-type]
    except (PredictionSourceError, OSError):
        return False
    return True


def render_footpass_analysis(split: str) -> None:
    """FOOTPASS Attack-v2 analysis on one dataset split (``train`` or ``val``)."""

    if split == "train":
        st.caption(
            "Phân tích các trận FOOTPASS split **train** (dữ liệu dùng để train model): "
            f"`playbyplay_GT/playbyplay_train.json` và `data/train_tactical_data.h5`."
        )
    else:
        st.caption(
            "Phân tích các trận FOOTPASS split **validation** (game 18 / 24 / 47), "
            "không dùng để train model."
        )
    mode = st.radio(
        "Evidence mode", ("oracle", "predicted"), horizontal=True, key=f"{split}_evidence_mode"
    )
    predicted_model = None
    event_model_name = None
    min_confidence = 0.0
    class_threshold_items: tuple[tuple[str, float], ...] = ()
    selected_source = None
    selected_source_path: Path | None = None
    if mode == "predicted":
        registered_sources = []
        if split == "val":
            try:
                registered_sources = list(prediction_source_choices())
            except PredictionSourceError as error:
                st.error(f"Prediction registry lỗi: {error}")
        else:
            st.caption(
                "Prediction registry chỉ đăng ký nguồn trên val. Split train đọc "
                f"`playbyplay_PRED/playbyplay_<model>_{split}.json` nếu đã infer."
            )
        if registered_sources:
            source_registry = load_prediction_registry()
            verified_indexes = [
                index
                for index, source in enumerate(registered_sources)
                if _prediction_file_ok(source_registry, source)
            ]
            default_source_index = 0 if 0 in verified_indexes or not verified_indexes else verified_indexes[0]
            selected_source = st.selectbox(
                "Prediction source",
                registered_sources,
                index=default_source_index,
                key=f"{split}_prediction_source",
                format_func=lambda item: (
                    f"{item.id} ({item.kind}, {item.role}"
                    + (f", F1 {item.micro_f1:.3f}" if item.micro_f1 is not None else "")
                    + ")"
                ),
                help=(
                    "Đọc từ models/prediction-sources.json. Nguồn primary là DST tốt nhất đã qua "
                    "gate micro-F1; TAAD thô chỉ là ablation."
                ),
            )
            predicted_model = selected_source.kind
            try:
                selected_source_path = verify_prediction_source(
                    load_prediction_registry(), selected_source
                )
                st.caption(
                    f"`{selected_source.id}` · sha256 `{selected_source.sha256[:12]}` · "
                    f"official micro-F1 {selected_source.micro_f1} (τ=0.15, δ=12)"
                )
            except PredictionSourceError as error:
                st.error(str(error))
        else:
            predicted_model = st.selectbox(
                "Action model",
                ("DST", "TAAD"),
                help="DST denoises the raw TAAD sequence and is the recommended predicted baseline.",
                key=f"{split}_action_model",
            )
        event_model_name = selected_source.id if selected_source is not None else predicted_model
        min_confidence = st.slider(
            "Action confidence threshold",
            0.0,
            1.0,
            0.15,
            0.05,
            help="0.15 is the official FOOTPASS evaluation threshold; use 0.60 for a precision-oriented ablation.",
            key=f"{split}_min_confidence",
        )
        if selected_source is not None:
            source_profile = selected_source.resolved_threshold_profile(load_prediction_registry().path)
        else:
            source_profile = DEFAULT_EVENT_PROFILE if predicted_model == "DST" else None
        if source_profile is not None and source_profile.is_file():
            use_calibrated_events = st.toggle(
                "Dùng class-aware event calibration",
                value=True,
                help=(
                    f"Profile `{source_profile.name}` được fit cho đúng nguồn này "
                    "(tuning game 18, held-out game 24/47)."
                ),
                key=f"{split}_class_aware",
            )
            if use_calibrated_events:
                class_threshold_items = tuple(sorted(load_event_thresholds(source_profile).items()))
                event_model_name = f"{event_model_name}_calibrated"
        elif predicted_model == "DST":
            st.caption(
                "Nguồn DST này chưa có threshold profile riêng; chạy calibrate-footpass-events trước "
                "khi bật class-aware calibration."
            )
        if predicted_model == "TAAD":
            st.caption("Ablation only: TAAD thô không thay DST trên đường Attack-v2.")
    if mode == "oracle":
        default_playbyplay = DEFAULT_FOOTPASS / "playbyplay_GT" / f"playbyplay_{split}.json"
    elif selected_source_path is not None:
        default_playbyplay = selected_source_path
    else:
        default_playbyplay = (
            DEFAULT_FOOTPASS / "playbyplay_PRED" / f"playbyplay_{predicted_model}_{split}.json"
        )
    playbyplay = Path(
        st.text_input(
            "Play-by-play JSON",
            str(default_playbyplay),
            key=f"{split}_playbyplay_{mode}_{event_model_name or 'oracle'}",
        )
    )
    tactical = Path(
        st.text_input(
            "Tactical HDF5",
            str(DEFAULT_FOOTPASS / "data" / f"{split}_tactical_data.h5"),
            key=f"{split}_tactical_h5",
        )
    )
    xg_model = DEFAULT_XG_V2_MODEL if DEFAULT_XG_V2_MODEL.is_file() else None
    if xg_model is not None:
        st.caption(
            "Chuỗi kết thúc bằng event `shot` được chấm xG-v2 train trên StatsBomb open data "
            f"(`{xg_model.name}`)."
        )
    else:
        st.error(
            f"Thiếu `{DEFAULT_XG_V2_MODEL}`. Phân tích có event cuối là shot sẽ dừng vì không có xG."
        )
    max_gap = st.slider(
        "Observation gap tạo censoring (seconds)",
        4.0,
        15.0,
        8.0,
        1.0,
        help=(
            "Khoảng im lặng dài hơn mức này thì possession bị censor, không bịa kết thúc. "
            "Pha đá bồi của cùng một đội trong 3 giây vẫn được nối vào chuỗi vừa sút."
        ),
        key=f"{split}_max_gap",
    )

    if playbyplay.exists() and tactical.exists():
        keys = playbyplay_keys(str(playbyplay))
        match_key = st.selectbox("Match half", keys, key=f"{split}_match_half")
        analysis_data = load_attack_v2_analysis(
            str(playbyplay),
            match_key,
            mode,
            str(tactical) if tactical.exists() else None,
            max_gap,
            min_confidence,
            event_model_name,
            str(xg_model) if xg_model else None,
            xg_model.stat().st_mtime if xg_model else None,
            class_threshold_items,
        )
        possessions_data = tuple(analysis_data.get("possessions", ()))
        attacks_data = tuple(analysis_data.get("attacks", ()))
        summary = dict(analysis_data.get("summary") or {})
        xg_payload = dict(analysis_data.get("xg") or {})
        xg_records = [
            item
            for item in xg_payload.get("records") or ()
            if isinstance(item, dict) and item.get("possession_id")
        ]
        xg_by_action = {
            str(item.get("action_id")): item
            for item in xg_records
            if item.get("action_id")
        }
        xg_by_possession: dict[str, dict] = {}
        for item in xg_records:
            possession_id = str(item.get("possession_id"))
            current = xg_by_possession.get(possession_id)
            if current is None or item.get("terminal_shot"):
                xg_by_possession[possession_id] = item
        source_tier = str(analysis_data.get("source_tier", "unknown"))
        metric_columns = st.columns(5)
        metric_columns[0].metric("Observed actions", len(analysis_data.get("observed_actions", ())))
        metric_columns[1].metric("Possessions", len(possessions_data))
        metric_columns[2].metric("Strict Attack-v2", len(attacks_data))
        metric_columns[3].metric("Replay excluded", summary.get("excluded_replay_actions", 0))
        metric_columns[4].metric("Shot có xG", int(xg_payload.get("shot_terminals_scored") or 0))
        st.caption(f"Source tier: `{source_tier}`")
        if xg_payload.get("model_id"):
            st.caption(f"xG model `{xg_payload.get('model_id')}` · StatsBomb open data")

        gold_index = load_gold_outcome_index()
        observed_actions_data = tuple(analysis_data.get("observed_actions", ()))
        excluded_replay_ids = tuple(analysis_data.get("excluded_replay_action_ids", ()))
        shot_insights = build_shot_insights(
            observed_actions_data,
            possessions_data,
            attacks_data,
            excluded_replay_action_ids=excluded_replay_ids,
            gold_index=gold_index,
        )
        with st.expander("Audit shot → Attack-v2", expanded=False):
            render_shot_promotion_audit(shot_insights)

        goal_truth = scoreboard_goal_index().for_match_key(match_key)
        scoreboard_outcomes = scoreboard_possession_outcomes(
            match_key,
            tuple(ObservedActionV2.from_dict(item) for item in observed_actions_data),
            tuple(PossessionSequenceV2.from_dict(item) for item in possessions_data),
            scoreboard_goal_index(),
        )
        goal_possession_ids = {
            possession_id
            for possession_id, (outcome, _source) in scoreboard_outcomes.items()
            if outcome == "goal"
        }
        attack_possession_ids = {item["possession"]["possession_id"] for item in attacks_data}
        if goal_truth is None:
            st.caption("Goal truth: chưa có scoreboard cho trận này (`attack-analysis build-scoreboard-goals`).")
        else:
            goal_source = (
                "scoreboard trong video"
                if goal_truth.source == "video_scoreboard"
                else "suy từ event: dừng ≥45 s rồi đội kia giao bóng ở phần sân nhà (feed không đọc được scoreboard)"
            )
            half_start = observed_actions_data[0]["timestamp_sec"] if observed_actions_data else 0.0
            half_end = observed_actions_data[-1]["timestamp_sec"] if observed_actions_data else 0.0
            half_goals = sum(
                half_start <= value <= half_end + SCORING_LOOKBACK_SEC
                for value in goal_truth.goal_times_sec
            )
            st.caption(
                f"Goal truth ({goal_source}): {half_goals} bàn trong hiệp · "
                f"{len(goal_possession_ids & attack_possession_ids)} nằm trong danh sách Attack-v2"
            )

        sequence_filter = st.radio(
            "Sequence filter",
            (
                SEQUENCE_FILTER_CONFIRMED,
                SEQUENCE_FILTER_GOALS,
                SEQUENCE_FILTER_CENSORED_SHOT,
                SEQUENCE_FILTER_ALL_POSSESSIONS,
            ),
            horizontal=True,
            key=f"{split}_sequence_filter",
        )
        selected_values = filter_sequence_entries(
            sequence_filter=(
                SEQUENCE_FILTER_CONFIRMED if sequence_filter == SEQUENCE_FILTER_GOALS else sequence_filter
            ),
            possessions=possessions_data,
            attacks=attacks_data,
            censored_possession_ids=shot_insights.censored_possession_ids,
        )
        if sequence_filter == SEQUENCE_FILTER_GOALS:
            selected_values = [
                (kind, item)
                for kind, item in selected_values
                if item["possession"]["possession_id"] in goal_possession_ids
            ]
            if not selected_values:
                st.info("Không có bàn thắng nào trong hiệp này theo goal truth.")
        if sequence_filter == SEQUENCE_FILTER_CONFIRMED and not selected_values:
            st.warning(
                "Không có terminal allowlist hoàn chỉnh ở operating point này; thử filter khác."
            )
        elif sequence_filter == SEQUENCE_FILTER_CENSORED_SHOT and not selected_values:
            st.info("Không có possession shot terminal bị censor trong hiệp này.")

        if selected_values:
            def v2_sequence_label(index: int) -> str:
                kind, item = selected_values[index]
                possession_dict = item["possession"] if kind == "attack" else item
                possession = PossessionSequenceV2.from_dict(possession_dict)
                terminal = possession.terminal
                terminal_name = terminal.terminal_type.value if terminal is not None else "no terminal"
                identifier = item.get("attack_id") if kind == "attack" else possession.possession_id
                shot_outcome, _ = resolve_shot_outcome(
                    possession,
                    gold_index=gold_index,
                    scoreboard_outcomes=scoreboard_outcomes,
                )
                outcome_suffix = (
                    f" · {format_shot_outcome(shot_outcome)}" if shot_outcome else ""
                )
                goal_prefix = "GOAL · " if shot_outcome == "goal" else ""
                ordered_ids = [action.action_id for action in possession.actions]
                xg_values = [
                    float(item["xg_probability"])
                    for action_id in ordered_ids
                    for item in xg_records
                    if item.get("action_id") == action_id
                    and item.get("xg_probability") is not None
                ]
                if len(xg_values) > 1:
                    xg_suffix = " · xG " + " → ".join(f"{value:.3f}" for value in xg_values)
                elif xg_values:
                    xg_suffix = f" · xG {xg_values[0]:.3f}"
                else:
                    xg_suffix = ""
                return (
                    f"{goal_prefix}{'ATTACK' if kind == 'attack' else possession.status.value.upper()} · "
                    f"{identifier} · {possession.start_sec:.1f}-{possession.end_sec:.1f}s · "
                    f"{terminal_name}{outcome_suffix}{xg_suffix}"
                )

            selected_index = st.selectbox(
                "Attack/Possession sequence",
                tuple(range(len(selected_values))),
                format_func=v2_sequence_label,
                key=f"{split}_sequence_{match_key}_{sequence_filter}",
            )
            selected_kind, selected_payload = selected_values[selected_index]
            attack_v2 = (
                AttackSequenceV2.from_dict(selected_payload)
                if selected_kind == "attack"
                else None
            )
            possession_v2 = (
                attack_v2.possession
                if attack_v2 is not None
                else PossessionSequenceV2.from_dict(selected_payload)
            )
            terminal = possession_v2.terminal
            shot_outcome, shot_outcome_source = resolve_shot_outcome(
                possession_v2,
                gold_index=gold_index,
                scoreboard_outcomes=scoreboard_outcomes,
            )
            detail_columns = st.columns(6)
            detail_columns[0].metric("Status", possession_v2.status.value)
            detail_columns[1].metric("Stable team", possession_v2.possession_team_id)
            detail_columns[2].metric("Actions", len(possession_v2.actions))
            detail_columns[3].metric("Duration", f"{possession_v2.end_sec - possession_v2.start_sec:.1f}s")
            detail_columns[4].metric(
                "Terminal", terminal.terminal_type.value if terminal is not None else "unknown"
            )
            detail_columns[5].metric("Shot outcome", format_shot_outcome(shot_outcome))
            if shot_outcome_source:
                st.caption(f"Outcome source: `{shot_outcome_source}`")
            shot_xg = xg_by_possession.get(possession_v2.possession_id)
            if is_shot_event(ending_event_action(possession_v2)):
                if shot_xg is None or shot_xg.get("xg_probability") is None:
                    st.error("Chuỗi kết thúc bằng event shot nhưng không có chỉ số xG StatsBomb.")
                else:
                    st.metric("xG", f"{float(shot_xg['xg_probability']):.3f}")
                    st.caption(
                        "xG-v2 train trên StatsBomb open data · "
                        f"`{shot_xg.get('model_id')}` · "
                        f"{'Attack-v2' if shot_xg.get('confirmed_attack') else 'chuỗi chưa complete'}"
                    )

            timeline_rows = [
                {
                    "Frame": action.frame,
                    "Time": round(action.timestamp_sec, 2),
                    "Action": ACTION_LABELS.get(action.action_type, action.action_type),
                    "Outcome": action.outcome or "—",
                    "Actor team": action.actor_team_id or "unknown",
                    "Possession team": action.possession_team_id or "unknown",
                    "L2R": action.raw_left_to_right,
                    "x": None if action.x is None else round(action.x, 1),
                    "x_attack": (
                        None
                        if possession_relative_position(
                            action.x,
                            action.y,
                            actor_team_id=action.actor_team_id,
                            possession_team_id=action.possession_team_id,
                        )[0]
                        is None
                        else round(
                            possession_relative_position(
                                action.x,
                                action.y,
                                actor_team_id=action.actor_team_id,
                                possession_team_id=action.possession_team_id,
                            )[0],
                            1,
                        )
                    ),
                    "Jersey": action.jersey,
                    "BBox": action.bbox_visible,
                    "Replay": action.replay,
                    "P(action)": action.action_probability,
                    "xG": (
                        f"{float(xg_by_action[action.action_id]['xg_probability']):.3f}"
                        if action.action_id in xg_by_action
                        and xg_by_action[action.action_id].get("xg_probability") is not None
                        else "—"
                    ),
                    "Evidence": ", ".join(action.evidence_ids),
                }
                for action in possession_v2.actions
            ]
            st.dataframe(pd.DataFrame(timeline_rows), width="stretch", hide_index=True)
            median_attack_x = attack_view_median_x(possession_v2.actions)
            if median_attack_x is not None and median_attack_x < 52.5:
                st.warning(
                    f"Median x_attack_view của đội kiểm soát = {median_attack_x:.1f} m "
                    "(< 52.5). Có thể direction FOOTPASS bị đảo."
                )
            pitch_actions = actions_with_resolved_outcome(
                possession_v2,
                shot_outcome=shot_outcome,
                outcome_source=shot_outcome_source,
            )
            show_event_pitch(
                pitch_actions,
                heading="Bản đồ event trên sân",
                caption=(
                    "Tọa độ possession-relative: mọi điểm cùng hướng tấn công của đội kiểm soát "
                    "(x → 105). Viền cam = hành động đối phương. "
                    "Hover từng điểm để xem kết quả (gold/terminal/action) và provenance."
                ),
                attack_relative=True,
            )

            video_resolver = load_video_path_resolver()
            yolo_video_path = video_resolver.yolo_video_path(match_key)
            preview_video_path = video_resolver.preview_path(match_key)
            preview_sequence = possession_to_preview_sequence(
                possession_v2,
                sequence_id=(
                    attack_v2.attack_id
                    if attack_v2 is not None
                    else possession_v2.possession_id
                ),
                source=f"footpass_{mode}" if split == "val" else f"footpass_{split}_{mode}",
            )
            st.markdown("#### Preview video sự kiện")
            if video_resolver.dual_video_enabled(match_key):
                st.caption(
                    "Preview và overlay YOLO đều dùng Full HD "
                    f"({preview_video_path.name}). Mapping thời gian: "
                    f"`{video_resolver.sync_path or 'offset mặc định 0s'}`."
                )
            preview_start_sec = video_resolver.match_sec_to_preview_sec(
                match_key,
                possession_v2.start_sec,
            )
            if preview_video_path.is_file():
                st.video(str(preview_video_path), start_time=max(0, int(preview_start_sec)))
                st.caption(
                    video_resolver.describe_preview(
                        match_key,
                        start_sec=possession_v2.start_sec,
                        end_sec=possession_v2.end_sec,
                    )
                )
            else:
                st.warning(f"Không tìm thấy video preview: {preview_video_path}")

            render_calibrated_window(
                preview_sequence,
                match_key=match_key,
                video_path=yolo_video_path,
                playbyplay_path=playbyplay,
                tactical_path=tactical,
                video_resolver=video_resolver,
            )

            conclusion_key = f"conclusion_{preview_sequence.attack_id}"
            window_manifest = st.session_state.get(f"window_manifest_{preview_sequence.attack_id}")
            can_conclude = attack_v2 is not None
            if openai_api_key_configured():
                st.caption(f"Kết luận LLM dùng `{openai_model()}` (model rẻ nhất trong allowlist).")
            if not DEFAULT_XG_V2_MODEL.is_file():
                st.caption(f"Thiếu `{DEFAULT_XG_V2_MODEL}` — xG sẽ abstain cho đến khi có artifact local.")
            if not window_manifest:
                st.caption("Chạy YOLO trên cửa sổ sự kiện để fusion spatial features vào xG-v2.")
            if st.button(
                "Đưa ra kết luận",
                type="primary",
                key=f"conclude_{preview_sequence.attack_id}",
                disabled=not can_conclude,
            ):
                if attack_v2 is None:
                    st.session_state[conclusion_key] = {
                        "overview": None,
                        "verdict": insufficient_evidence_verdict(
                            possession_v2.possession_id,
                            (
                                "Chuỗi này chưa phải Attack-v2 (thiếu terminal shot/header/block hợp lệ).",
                            ),
                        ),
                        "evidence": None,
                        "conclusion": None,
                    }
                else:
                    store = load_benchmark_store()
                    role_benchmarks = store.get_role_benchmark("central_forward", "xg")
                    web_payload = {"enabled": False, "results": []}
                    if brave_search_configured():
                        query = (
                            f"football {attack_v2.terminal.terminal_type.value} chance quality "
                            f"{attack_v2.match_id}"
                        )
                        web_payload = brave_search_context(query)
                    web_results = tuple(dict(item) for item in web_payload.get("results") or ())
                    evidence = build_attack_evidence(
                        attack_v2,
                        window_manifest=window_manifest,
                        xg_model_path=DEFAULT_XG_V2_MODEL if DEFAULT_XG_V2_MODEL.is_file() else None,
                        team_mapping=default_team_mapping(attack_v2),
                        allow_research_only=True,
                        benchmarks=role_benchmarks,
                        web_context=web_results,
                        tactical_h5=tactical if tactical.exists() else None,
                    )
                    overview = attack_v2_research_overview(
                        attack_v2,
                        xg_record=evidence.xg_record,
                        benchmarks=store,
                    )
                    legacy_bundle = evidence.as_legacy_bundle(
                        shot_outcome=shot_outcome,
                        outcome_source=shot_outcome_source,
                    )
                    tools = EvidenceToolRegistry(
                        {attack_v2.attack_id: legacy_bundle},
                        benchmarks=store,
                        web_search=brave_search_context,
                    )
                    try:
                        if openai_api_key_configured():
                            client: DeterministicEvidenceClient | OpenAIResponsesClient = (
                                OpenAIResponsesClient(
                                    model=openai_model(),
                                    project_root=PROJECT_ROOT,
                                    response_language="Vietnamese",
                                )
                            )
                        else:
                            client = DeterministicEvidenceClient()
                        conclusion_v2 = client.evaluate_v2(legacy_bundle, tools)
                    except Exception as error:
                        LOGGER.exception("Conclusion evaluation failed")
                        st.error(describe_openai_error(error) if openai_api_key_configured() else str(error))
                        conclusion_v2 = DeterministicEvidenceClient().evaluate_v2(legacy_bundle, tools)
                    st.session_state[conclusion_key] = {
                        "overview": overview,
                        "evidence": evidence,
                        "conclusion": conclusion_v2,
                        "verdict": None,
                    }

            conclusion = st.session_state.get(conclusion_key)
            if conclusion:
                stored_conclusion = conclusion.get("conclusion")
                stored_evidence = conclusion.get("evidence")
                overview = conclusion.get("overview")
                if stored_conclusion is not None and stored_evidence is not None:
                    render_conclusion_v2(stored_conclusion, stored_evidence, overview if isinstance(overview, dict) else None)
                elif conclusion.get("verdict") is not None:
                    verdict = conclusion["verdict"]
                    label, icon = VERDICT_LABELS[verdict.label]
                    st.markdown(f"### {icon} Kết luận: {label}")
                    for limitation in verdict.limitations:
                        st.write(f"• {limitation}")

            if attack_v2 is not None:
                st.success(
                    "Confirmed Attack-v2: possession complete và kết thúc bằng terminal allowlist."
                )
            else:
                st.info(
                    "Đây là possession/threat research-only, không phải attack; "
                    "nút kết luận chỉ bật khi có terminal hợp lệ."
                )

            with st.expander("Field-level provenance và state transitions"):
                st.json(possession_v2.to_dict(), expanded=False)

        legacy = dict(analysis_data.get("legacy_compatibility") or {})
        with st.expander("V1 compatibility ablation (không phải attack)"):
            st.warning(
                "Các đoạn dưới đây chỉ là `PossessionCandidateV1/legacy_unvalidated`; "
                "không được dùng cho headline metric hay effectiveness."
            )
            st.json(legacy.get("summary", {}), expanded=False)
    else:
        if not playbyplay.exists():
            st.warning("Play-by-play file was not found.")
        if not tactical.exists():
            st.warning(
                "Tactical HDF5 is required for oracle/predicted assisted source tiers and stable team identity."
            )


analysis_tab, validation_tab, upload_tab, annotation_tab = st.tabs(
    ["Analysis", "Validation Analysis", "Video upload", "Annotation"]
)

with analysis_tab:
    render_footpass_analysis("train")

with validation_tab:
    render_footpass_analysis("val")


def conclude_attack_v2(attack: AttackSequenceV2, *, frames=None, shot_outcome: dict | None = None):
    """Same research conclusion as the FOOTPASS tab: xG, chance quality, tactical read."""

    store = load_benchmark_store()
    role_benchmarks = store.get_role_benchmark("central_forward", "xg")
    web_results: tuple[dict[str, object], ...] = ()
    if brave_search_configured():
        query = (
            f"football {attack.terminal.terminal_type.value} chance quality {attack.match_id}"
        )
        web_payload = brave_search_context(query)
        web_results = tuple(dict(item) for item in web_payload.get("results") or ())
    evidence = build_attack_evidence(
        attack,
        frames=frames,
        xg_model_path=DEFAULT_XG_V2_MODEL if DEFAULT_XG_V2_MODEL.is_file() else None,
        team_mapping=default_team_mapping(attack),
        allow_research_only=True,
        benchmarks=role_benchmarks,
        web_context=web_results,
    )
    overview = attack_v2_research_overview(
        attack,
        xg_record=evidence.xg_record,
        benchmarks=store,
    )
    legacy_bundle = evidence.as_legacy_bundle(
        shot_outcome=str(shot_outcome.get("outcome")) if shot_outcome else None,
        outcome_source="video_scoreboard_estimate" if shot_outcome else None,
    )
    if shot_outcome:
        label = VIDEO_SHOT_OUTCOME_LABELS.get(str(shot_outcome.get("outcome")), "Chưa rõ")
        legacy_bundle = dataclass_replace(
            legacy_bundle,
            limitations=(
                *legacy_bundle.limitations,
                f"Kết quả thực tế của cú dứt điểm (ước tính từ video, không phải nhãn gold): {label}; "
                f"căn cứ: {shot_outcome.get('summary')}. Scoreboard không gắn được với team_a/team_b, "
                "bàn thắng được quy cho đội đang tấn công. Đánh giá cần xét cả xG lẫn kết quả này.",
            ),
        )
    tools = EvidenceToolRegistry(
        {attack.attack_id: legacy_bundle},
        benchmarks=store,
        web_search=brave_search_context,
    )
    try:
        if openai_api_key_configured():
            client: DeterministicEvidenceClient | OpenAIResponsesClient = OpenAIResponsesClient(
                model=openai_model(),
                project_root=PROJECT_ROOT,
                response_language="Vietnamese",
            )
        else:
            client = DeterministicEvidenceClient()
        conclusion = client.evaluate_v2(legacy_bundle, tools)
    except Exception as error:
        LOGGER.exception("Conclusion evaluation failed")
        st.error(describe_openai_error(error) if openai_api_key_configured() else str(error))
        conclusion = DeterministicEvidenceClient().evaluate_v2(legacy_bundle, tools)
    return overview, evidence, conclusion


def _upload_attack_label(attack: AttackSequenceV2) -> str:
    possession = attack.possession
    terminal = possession.terminal.terminal_type.value if possession.terminal else "no terminal"
    start_clock, _spoken = _video_clock_parts(possession.start_sec)
    end_clock, _end_spoken = _video_clock_parts(possession.end_sec)
    return (
        f"ATTACK · {start_clock}–{end_clock} · {terminal} · "
        f"{len(possession.actions)} sự kiện · {attack.team_id}"
    )


def upload_event_views(extracted: dict[str, object]) -> dict[str, dict[str, object]]:
    """Event sources available after Phase 1, keyed like ``UPLOAD_EVENT_SOURCES``."""

    shared = {
        key: extracted.get(key)
        for key in (
            "observations_path",
            "annotated_video",
            "match_id",
            "scoreboard_reads",
            "scoreboard_error",
        )
    }
    views = {str(extracted.get("method") or "footpass_taad"): extracted}
    dst = extracted.get("dst")
    if isinstance(dst, dict):
        views["footpass_dst"] = {**dst, **shared}
    return views


def read_scoreboard_with_progress(video_path: Path, cache_path: Path | None):
    """Score bug readings every second (RapidOCR on CPU), cached next to the observations."""

    progress = st.progress(0.0, text="Đọc tỷ số trên scoreboard")
    try:
        readings = ScoreboardReader().read_video(
            video_path,
            cache_path=cache_path,
            progress=lambda done, total: progress.progress(
                min(1.0, done / max(1, total)),
                text=f"Đọc tỷ số trên scoreboard: {done}/{total} giây video",
            ),
        )
        return readings, None
    except (ImportError, OSError, RuntimeError) as error:
        LOGGER.warning("Scoreboard OCR unavailable: %s", error)
        return (), str(error)
    finally:
        progress.empty()


def apply_video_outcomes(attacks, readings, frames):
    """Attacks with the video-estimated shot outcome attached, plus evidence per attack id."""

    attacks = tuple(attacks or ())
    if not attacks:
        return attacks, {}
    outcomes = infer_attack_outcomes(attacks, readings, tuple(frames or ()))
    enriched = tuple(
        attack_with_video_outcome(attack, outcomes[attack.attack_id]) for attack in attacks
    )
    return enriched, {key: value.to_dict() for key, value in outcomes.items()}


def _shot_outcome_text(evidence: dict | None) -> str:
    if not evidence:
        return "—"
    return VIDEO_SHOT_OUTCOME_LABELS.get(str(evidence.get("outcome")), "Chưa rõ")


def _shot_outcome_basis(evidence: dict | None) -> str:
    if not evidence:
        return "—"
    source = SHOT_OUTCOME_SOURCES.get(str(evidence.get("source") or ""))
    summary = str(evidence.get("summary") or "")
    return f"{source}: {summary}" if source else summary


def _outcome_near(outcomes: dict, timestamp_sec: float) -> dict | None:
    best = min(
        outcomes.values(),
        key=lambda item: abs(float(item.get("shot_sec") or 0.0) - timestamp_sec),
        default=None,
    )
    if best is None or abs(float(best.get("shot_sec") or 0.0) - timestamp_sec) > 0.5:
        return None
    return best


def render_extracted_events(extracted: dict[str, object]) -> None:
    """Phase 1 result: calibration, then TAAD and DST event lists side by side."""

    calibration = extracted.get("calibration") or {}
    if calibration:
        st.caption(
            f"Tọa độ sân: hiệu chỉnh từ điểm mốc sân (model {calibration.get('model')}). "
            f"{float(calibration.get('coverage') or 0.0):.0%} frame có tọa độ, "
            f"{int(calibration.get('direct_fits') or 0)}/{int(calibration.get('frames') or 0)} frame fit trực tiếp, "
            f"trung vị {calibration.get('median_inliers')} điểm mốc, sai số "
            f"{float(calibration.get('median_rmse_m') or 0.0):.2f} m. Frame không fit được để trống tọa độ "
            "thay vì đoán. Video YOLO vẽ lại đường kẻ sân đã hiệu chỉnh (màu hồng) để kiểm tra bằng mắt."
        )
    views = upload_event_views(extracted)
    if "footpass_dst" not in views:
        if extracted.get("dst_error"):
            st.warning(f"Không chạy được FOOTPASS DST: {extracted['dst_error']}")
        _render_event_source(extracted)
        return
    tabs = st.tabs([UPLOAD_EVENT_SOURCES.get(key, key) for key in views])
    for tab, view in zip(tabs, views.values()):
        with tab:
            _render_event_source(view)


def _render_event_source(extracted: dict[str, object]) -> None:
    events = tuple(extracted.get("events") or ())
    attacks = tuple(extracted.get("attacks") or ())
    highlights = tuple(extracted.get("highlights") or ())
    if extracted.get("method") == "footpass_taad":
        st.caption(
            f"{len(highlights)} đoạn camera toàn cảnh · {len(events)} sự kiện · "
            f"{len(attacks)} chuỗi tấn công. Sự kiện do FOOTPASS TAAD (checkpoint H100) "
            "nhận trên track cầu thủ YOLO. Cận cảnh, ăn mừng và replay bị loại vì TAAD chỉ học camera chính. "
            "Đội của mọi sự kiện trong một đoạn là đội thắng phiếu theo điểm TAAD, vì màu áo YOLO hay nhảy."
        )
    elif extracted.get("method") == "footpass_dst":
        st.caption(
            f"{len(highlights)} đoạn · {len(events)} sự kiện · {len(attacks)} chuỗi tấn công. "
            "FOOTPASS DST (checkpoint DST_Baseline_h100, ablation chưa promote, research only) đọc lại "
            "logits TAAD cùng vị trí, vận tốc cầu thủ trên sân đã hiệu chỉnh. "
            f"{int(extracted.get('assigned_tracks') or 0)} track YOLO được gán vai trò FOOTPASS theo vị trí "
            "trung bình. DST được train khi biết vị trí đủ 22 cầu thủ; video chỉ thấy một phần, người ngoài "
            "khung hình để giá trị thiếu (-15), nên DST ở đây yếu hơn số đo trên FOOTPASS val."
        )
    else:
        st.caption(
            f"{len(events)} sự kiện · {extracted.get('possession_count', 0)} possession · "
            f"{len(attacks)} chuỗi tấn công. Suy luận từ quỹ đạo bóng (không có FOOTPASS TAAD)."
        )
    shot_outcomes = dict(extracted.get("shot_outcomes") or {})
    if extracted.get("scoreboard_error"):
        st.warning(
            f"Không đọc được scoreboard ({extracted['scoreboard_error']}). Kết quả dứt điểm chỉ dựa vào bóng YOLO."
        )
    elif attacks:
        st.caption(
            f"Kết quả dứt điểm ước tính từ video, không phải nhãn gold: đọc tỷ số scoreboard mỗi giây "
            f"({int(extracted.get('scoreboard_reads') or 0)} lần đọc được) và tìm bóng YOLO sau vạch "
            "giữa hai cột dọc trong 3 giây sau cú sút. Bàn thắng khi tổng tỷ số tăng 1 sau cú sút "
            "(trước cú sút kế tiếp); không thành bàn khi tỷ số vẫn giữ ít nhất 10 giây sau cú sút. "
            "Scoreboard không gắn được với team_a/team_b nên bàn thắng được quy cho đội đang tấn công."
        )
    if highlights:
        segment_rows = []
        model_name = "DST" if extracted.get("method") == "footpass_dst" else "TAAD"
        for highlight in highlights:
            terminal_outcome = (
                _outcome_near(shot_outcomes, float(highlight.terminal.timestamp_sec))
                if highlight.terminal is not None
                else None
            )
            start_clock, _spoken = _video_clock_parts(highlight.segment.start_sec)
            end_clock, _end_spoken = _video_clock_parts(highlight.segment.end_sec)
            terminal = highlight.terminal
            direction = getattr(highlight, "left_to_right", None)
            segment_rows.append(
                {
                    "Đoạn toàn cảnh": f"{start_clock}–{end_clock}",
                    "Đội tấn công": highlight.team or "—",
                    "Hướng tấn công": (
                        "—"
                        if direction is None
                        else "→ khung thành phải"
                        if direction
                        else "← khung thành trái"
                    ),
                    "Kết thúc": (
                        (
                            f"Dứt điểm ({model_name} nhận là chặn bóng)"
                            if terminal.action == "block"
                            else ACTION_LABELS.get(terminal.action, terminal.action)
                        )
                        + f" {terminal.timestamp_sec:.2f}s ({model_name} {terminal.score:.2f})"
                        if terminal is not None
                        else "Không thấy cú sút/đánh đầu"
                    ),
                    "Kết quả (ước tính video)": _shot_outcome_text(terminal_outcome),
                    "Số sự kiện": len(highlight.events),
                }
            )
        st.dataframe(pd.DataFrame(segment_rows), width="stretch", hide_index=True)
    if not events:
        st.warning("Không suy ra sự kiện. Cần cầu thủ, bóng và tọa độ sân.")
        return
    event_rows = []
    for event in events:
        clock, _spoken = _video_clock_parts(float(event.timestamp))
        event_rows.append(
            {
                "Thời gian": f"{clock} ({event.timestamp:.2f}s)",
                "Sự kiện": ACTION_LABELS.get(event.event_type, event.event_type),
                "Đội": event.team,
                "Track": event.player_id or "—",
                "Điểm model": (
                    f"{event.confidence:.2f}"
                    if event.source in {"footpass_taad", "footpass_dst"}
                    else "—"
                ),
                "Tọa độ": (
                    f"({event.x:.1f}, {event.y:.1f})"
                    if event.x is not None and event.y is not None
                    else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)
    if not attacks:
        st.info(
            "Chưa có chuỗi tấn công. Một attack cần kiểm soát bóng rồi kết thúc bằng cú sút hoặc đánh đầu."
        )
        return
    attack_rows = []
    for attack in attacks:
        possession = attack.possession
        start_clock, _spoken = _video_clock_parts(possession.start_sec)
        end_clock, _end_spoken = _video_clock_parts(possession.end_sec)
        attack_rows.append(
            {
                "Chuỗi": _upload_attack_label(attack),
                "Bắt đầu": start_clock,
                "Kết thúc": end_clock,
                "Đội": attack.team_id,
                "Kết quả dứt điểm": _shot_outcome_text(shot_outcomes.get(attack.attack_id)),
                "Căn cứ": _shot_outcome_basis(shot_outcomes.get(attack.attack_id)),
            }
        )
    st.dataframe(pd.DataFrame(attack_rows), width="stretch", hide_index=True)


def render_upload_analysis(extracted: dict[str, object], *, video_path: Path) -> None:
    """Phase 2: YOLO overlay, xG-v2 and the AI read for one extracted attack."""

    views = upload_event_views(extracted)
    if len(views) > 1:
        source = st.radio(
            "Nguồn chuỗi tấn công",
            tuple(views),
            format_func=lambda key: UPLOAD_EVENT_SOURCES.get(key, key),
            horizontal=True,
            key=f"upload_source_{extracted.get('match_id')}",
        )
        extracted = {**views[source], "match_id": f"{extracted.get('match_id')}-{source}"}
    attacks = tuple(extracted.get("attacks") or ())
    if not attacks:
        st.info(
            "Phase 1 chưa có chuỗi tấn công. YOLO, AI và model xG chỉ chạy trên một chuỗi đã trích."
        )
        return

    selected_index = st.selectbox(
        "Chuỗi sự kiện tấn công",
        tuple(range(len(attacks))),
        format_func=lambda index: _upload_attack_label(attacks[index]),
        key=f"upload_attack_{extracted.get('match_id')}",
    )
    attack = attacks[selected_index]
    if st.button(
        "Phân tích với YOLO, AI và xG",
        type="primary",
        key=f"analyze_{attack.attack_id}",
    ):
        with st.spinner("Đang phân tích: overlay YOLO, model xG-v2 và nhận định AI..."):
            observations_path = Path(str(extracted.get("observations_path") or ""))
            frames = (
                read_observations_jsonl(observations_path) if observations_path.is_file() else None
            )
            overview, evidence, conclusion = conclude_attack_v2(
                attack,
                frames=frames,
                shot_outcome=dict(extracted.get("shot_outcomes") or {}).get(attack.attack_id),
            )
        st.session_state[f"upload_conclusion_{extracted.get('match_id')}"] = {
            "attack_id": attack.attack_id,
            "overview": overview,
            "evidence": evidence,
            "conclusion": conclusion,
        }

    stored = st.session_state.get(f"upload_conclusion_{extracted.get('match_id')}")
    if not stored or stored.get("attack_id") != attack.attack_id:
        return
    annotated_path = Path(str(extracted.get("annotated_video") or ""))
    if annotated_path.is_file():
        show_yolo_overlay(
            annotated_path,
            heading="YOLO",
            start_sec=max(0.0, attack.possession.start_sec - UPLOAD_CLIP_LEAD_SEC),
            end_sec=attack.possession.end_sec + UPLOAD_CLIP_TAIL_SEC,
        )
    elif video_path.is_file():
        show_yolo_overlay(
            video_path,
            heading="Video gốc (không có overlay YOLO)",
            start_sec=max(0.0, attack.possession.start_sec - UPLOAD_CLIP_LEAD_SEC),
            end_sec=attack.possession.end_sec + UPLOAD_CLIP_TAIL_SEC,
            clip_dir=PROJECT_ROOT / "runs" / "upload-cache" / "clips",
        )
    shot_outcome = dict(extracted.get("shot_outcomes") or {}).get(attack.attack_id)
    if shot_outcome:
        outcome_text = _shot_outcome_text(shot_outcome)
        message = (
            f"Kết quả thực tế (ước tính từ video, không phải nhãn gold): **{outcome_text}**. "
            f"{_shot_outcome_basis(shot_outcome)}."
        )
        if shot_outcome.get("outcome") == "goal":
            st.success(message)
        elif shot_outcome.get("outcome") == "no_goal":
            st.info(message)
        else:
            st.warning(message)
    render_conclusion_v2(stored["conclusion"], stored["evidence"], stored["overview"])
    timeline_rows = []
    xg_by_action = {}
    evidence = stored["evidence"]
    if evidence.xg_record is not None and evidence.xg_record.terminal_id:
        xg_by_action[evidence.attack.terminal.action_id] = evidence.xg_record.xg_probability
    for action in attack.possession.actions:
        timeline_rows.append(
            {
                "Thời gian": f"{action.timestamp_sec:.2f}s",
                "Sự kiện": ACTION_LABELS.get(action.action_type, action.action_type),
                "Đội": action.actor_team_id or "—",
                "x": None if action.x is None else round(action.x, 1),
                "y": None if action.y is None else round(action.y, 1),
                "xG": (
                    f"{xg_by_action[action.action_id]:.3f}"
                    if action.action_id in xg_by_action
                    else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(timeline_rows), width="stretch", hide_index=True)
    show_event_pitch(
        attack.possession.actions,
        heading="Bản đồ đợt tấn công",
        caption="Tọa độ đã chiếu về hướng tấn công của đội đang kiểm soát.",
        attack_relative=True,
    )


with upload_tab:
    cache_dir = PROJECT_ROOT / "runs" / "upload-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    clip_path: Path | None = None
    digest: str | None = None
    uploaded = st.file_uploader(
        "Chọn video từ máy",
        type=("mp4", "mkv", "mov"),
        help="Video được lưu vào runs/upload-cache theo SHA-256 nội dung; chọn lại cùng file sẽ dùng lại cache.",
    )
    if uploaded is None:
        st.info("Chọn một video phát sóng (MP4/MKV/MOV) để bắt đầu.")
    else:
        content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        clip_path = cache_dir / f"{digest}{Path(uploaded.name).suffix.lower()}"
        if not clip_path.exists():
            clip_path.write_bytes(content)
        del content
    source_identity = digest

    if clip_path is not None and source_identity is not None:
        metadata: dict = {}
        try:
            metadata = inspect_video(clip_path).to_dict()
            st.video(str(clip_path))
            metadata_columns = st.columns(4)
            metadata_columns[0].metric("Thời lượng", f"{metadata['duration_sec']:.1f}s")
            metadata_columns[1].metric("FPS", f"{metadata['fps']:.1f}")
            metadata_columns[2].metric("Độ phân giải", f"{metadata['width']}×{metadata['height']}")
            metadata_columns[3].metric("Số frame", f"{metadata['frame_count']:,}")
        except Exception as error:
            st.error(str(error))

        with st.container(border=True):
            st.subheader("Phase 1 · Trích xuất sự kiện")
            st.caption(
                "Đọc video và lập danh sách sự kiện, possession và chuỗi tấn công. "
                "Kết quả của phase này là bảng sự kiện."
            )
            whole_video = st.checkbox(
                "Toàn bộ video",
                value=True,
                help="Bỏ chọn để chỉ xử lý một đoạn đầu, dùng khi muốn thử nhanh.",
            )
            max_frames = None
            if not whole_video:
                max_frames = st.slider(
                    "Số frame đọc khi trích sự kiện",
                    min_value=25,
                    max_value=max(25, min(750, int(metadata.get("frame_count") or 750))),
                    value=min(250, max(25, min(750, int(metadata.get("frame_count") or 750)))),
                    step=25,
                )
            elif int(metadata.get("frame_count") or 0) > 4000:
                st.caption(
                    f"Video có {int(metadata['frame_count']):,} frame. Trích toàn bộ có thể mất nhiều phút; "
                    "kết quả được cache theo file."
                )
            try:
                import torch

                gpu_ready = bool(torch.cuda.is_available())
            except ImportError:
                gpu_ready = False
            device = st.selectbox(
                "Thiết bị trích xuất",
                ("0", "cpu") if gpu_ready else ("cpu",),
                format_func=lambda value: "GPU CUDA" if value == "0" else "CPU",
            )
            use_landmarks = device != "cpu" and DEFAULT_PITCH_MODEL.is_file()
            vision_config = VisionConfig(
                pose_model=str(DEFAULT_POSE_MODEL),
                detection_model=str(DEFAULT_DETECTION_MODEL),
                device=device,
                image_size=960,
                max_frames=max_frames,
                field_lines=True,
                polygon_pitch_fallback=not use_landmarks,
            )
            if not use_landmarks:
                st.caption(
                    "Không có model điểm mốc sân hoặc đang chạy CPU: tọa độ sân lấy từ vùng cỏ "
                    "(ép vùng cỏ thấy được thành cả sân), vị trí cầu thủ sẽ sai."
                )
            extract_key = f"upload_extract_v3_{source_identity}_{max_frames}_{device}_{use_landmarks}"
            if st.button("Trích xuất sự kiện", type="primary"):
                try:
                    with st.spinner("Đang trích xuất sự kiện..."):
                        projector = None
                        calibration_summary: dict[str, object] = {}
                        cache_context: dict[str, object] = {}
                        if use_landmarks:
                            calibration_progress = st.progress(0.0, text="Hiệu chỉnh sân từ điểm mốc...")
                            calibration = load_or_calibrate_video(
                                clip_path,
                                cache_dir / "pitch",
                                PitchModelConfig(str(DEFAULT_PITCH_MODEL), device=f"cuda:{device}"),
                                video_digest=digest,
                                max_frames=max_frames,
                                progress=lambda done, total: calibration_progress.progress(
                                    min(1.0, done / max(1, total)),
                                    text=f"Hiệu chỉnh sân từ điểm mốc: {done}/{total} frame",
                                ),
                            )
                            calibration_progress.empty()
                            projector = KeypointPitchProjector(calibration)
                            calibration_summary = dict(calibration.summary)
                            cache_context = {"pitch_calibration": calibration.digest()}
                        manifest = run_cached_video_pipeline(
                            clip_path,
                            cache_dir / "vision",
                            digest,
                            vision_config,
                            projector=projector,
                            cache_context=cache_context,
                        )
                        observations_path = Path(str((manifest.get("outputs") or {}).get("observations") or ""))
                        frames = (
                            read_observations_jsonl(observations_path) if observations_path.is_file() else ()
                        )
                        match_id = f"upload-{(source_identity or 'clip')[:12]}"
                        fps = float((manifest.get("vision") or {}).get("fps") or metadata.get("fps") or 25.0)
                        method = "footpass_taad"
                        highlights = ()
                        dst_view: dict[str, object] | None = None
                        dst_error: str | None = None
                        try:
                            if device == "cpu":
                                raise RuntimeError("FOOTPASS TAAD cần GPU CUDA")
                            taad_result = extract_taad_highlights(
                                clip_path,
                                frames,
                                width=float(metadata.get("width") or 1920),
                                height=float(metadata.get("height") or 1080),
                                fps=fps,
                                match_id=match_id,
                                cache_path=observations_path.parent / "taad_detections.json",
                            )
                            highlights = taad_result.highlights
                            decoded = taad_result.decoded
                            events = tuple(
                                event for highlight in highlights for event in highlight.events
                            )
                            try:
                                dst_result = extract_dst_highlights(
                                    frames,
                                    taad_result.segments,
                                    taad_result.outputs,
                                    taad_result.highlights,
                                    match_id=match_id,
                                    fps=fps,
                                    device=f"cuda:{device}",
                                )
                                dst_view = {
                                    "method": "footpass_dst",
                                    "highlights": dst_result.highlights,
                                    "events": tuple(
                                        event
                                        for highlight in dst_result.highlights
                                        for event in highlight.events
                                    ),
                                    "attacks": dst_result.decoded.attacks,
                                    "possession_count": len(dst_result.decoded.possessions),
                                    "checkpoint": dst_result.checkpoint,
                                    "assigned_tracks": sum(
                                        len(items) for items in dst_result.assignments.values()
                                    ),
                                }
                            except (FileNotFoundError, RuntimeError, ImportError, ValueError) as error:
                                LOGGER.warning("DST unavailable: %s", error)
                                dst_error = str(error)
                        except (FileNotFoundError, RuntimeError, ImportError) as taad_error:
                            LOGGER.warning("TAAD unavailable, falling back to ball heuristics: %s", taad_error)
                            st.warning(f"Không chạy được FOOTPASS TAAD ({taad_error}). Dùng suy luận từ quỹ đạo bóng.")
                            method = "ball_heuristic"
                            events = infer_events_from_observations(frames) if frames else ()
                            _actions, decoded = sequences_from_video_events(
                                events,
                                match_id=match_id,
                                fps=fps,
                            )
                    readings, scoreboard_error = read_scoreboard_with_progress(
                        clip_path,
                        observations_path.parent / "scoreboard.json"
                        if observations_path.is_file()
                        else None,
                    )
                    attacks, shot_outcomes = apply_video_outcomes(decoded.attacks, readings, frames)
                    if dst_view is not None:
                        dst_attacks, dst_outcomes = apply_video_outcomes(
                            dst_view["attacks"], readings, frames
                        )
                        dst_view = {**dst_view, "attacks": dst_attacks, "shot_outcomes": dst_outcomes}
                    st.session_state[extract_key] = {
                        "method": method,
                        "highlights": highlights,
                        "events": events,
                        "attacks": attacks,
                        "shot_outcomes": shot_outcomes,
                        "scoreboard_reads": len(readings),
                        "scoreboard_error": scoreboard_error,
                        "possession_count": len(decoded.possessions),
                        "match_id": match_id,
                        "observations_path": str(observations_path),
                        "annotated_video": str((manifest.get("outputs") or {}).get("annotated_video") or ""),
                        "calibration": calibration_summary,
                        "dst": dst_view,
                        "dst_error": dst_error,
                    }
                except ValueError as error:
                    st.session_state.pop(extract_key, None)
                    st.warning(
                        "Chưa suy ra sự kiện. Cần cầu thủ, bóng và tọa độ sân "
                        f"(đường biên cỏ hoặc calibration). {error}"
                    )
                except Exception as error:
                    LOGGER.exception("Upload event extraction failed")
                    st.error(f"Trích xuất sự kiện thất bại: {error}")

            extracted = st.session_state.get(extract_key)
            if extracted:
                render_extracted_events(extracted)
            else:
                st.info("Bấm Trích xuất sự kiện để có danh sách sự kiện và các chuỗi tấn công của video.")

        with st.container(border=True):
            st.subheader("Phase 2 · Phân tích với YOLO, AI và xG")
            st.caption(
                "Một lượt bấm chạy overlay YOLO, model xG-v2 và nhận định AI trên chuỗi tấn công đã chọn."
            )
            if extracted:
                render_upload_analysis(extracted, video_path=clip_path)
            else:
                st.info("Hoàn thành Phase 1 trước. Phase 2 không tự chạy khi trích sự kiện.")

with annotation_tab:
    st.caption(
        "P11 còn mở: gán nhãn liên tục **development trước** (game_18 / 24 / 47), "
        "rồi train, **cuối cùng** mới seal test (game_0 / 36 / 50). "
        "Không mở frozen test để tune. Chạy `attack-analysis gold-status` và xem "
        "`docs/gold-annotation-runbook.md`."
    )
    manifest_file = st.file_uploader("Annotation manifest JSON", type="json", key="annotation_manifest")
    if manifest_file is not None:
        manifest = json.loads(manifest_file.getvalue())
        if manifest.get("schema_version") == "attack-gold-v2.0" and manifest.get("matches"):
            st.subheader("Attack-v2 continuous timeline annotation")
            st.warning(
                "Workflow này không hiển thị prediction/segment của mô hình. Gán nhãn trực tiếp từ "
                "timeline; game_0/36/50 được ghi vào final-test seal."
            )
            split_order = {"development": 0, "train": 1, "test": 2}
            matches = tuple(
                sorted(
                    manifest["matches"],
                    key=lambda item: (
                        split_order.get(str(item.get("split")), 9),
                        str(item.get("match_id")),
                    ),
                )
            )
            match_index = st.selectbox(
                "Match",
                range(len(matches)),
                format_func=lambda value: (
                    f"{matches[value]['match_id']} · {matches[value]['split']}"
                    + (" · PILOT" if matches[value].get("pilot") else "")
                ),
                key="gold_match",
            )
            match = matches[match_index]
            match_id = str(match["match_id"])
            if gold_split(match_id) == "test":
                st.error(
                    "Frozen test (game_0 / 36 / 50). Chỉ gán nhãn sau khi development "
                    "đã intra-rater và không dùng test để chọn threshold. Ghi vào sealed-test."
                )
            elif gold_split(match_id) == "development":
                st.info("Đúng thứ tự P11: đang gán nhãn development (18 / 24 / 47).")
            video_path = Path(str(match["video_path"]))
            clip_start = float(
                st.number_input(
                    "Video start (seconds)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key=f"gold_video_start_{match_id}",
                )
            )
            if video_path.is_file():
                st.video(str(video_path), start_time=int(clip_start))
            else:
                st.error(f"Không tìm thấy video: {video_path}")

            with st.form(f"gold_annotation_form_{match_id}", clear_on_submit=False):
                period = int(st.selectbox("Period", (1, 2)))
                stable_team_id = st.text_input("Stable team ID", value="team_a")
                start_sec = float(
                    st.number_input(
                        "Sequence start (s)", min_value=0.0, value=clip_start, step=0.04
                    )
                )
                end_sec = float(
                    st.number_input(
                        "Sequence end (s)",
                        min_value=0.0,
                        value=clip_start + 5.0,
                        step=0.04,
                    )
                )
                start_frame = int(
                    st.number_input(
                        "Start frame", min_value=0, value=int(round(start_sec * 25))
                    )
                )
                end_frame = int(
                    st.number_input(
                        "End frame",
                        min_value=0,
                        value=max(start_frame, int(round(end_sec * 25))),
                    )
                )
                label = st.selectbox("Sequence label", ANNOTATION_LABELS)
                terminal_type = st.selectbox("Terminal", TERMINAL_TYPES)
                completeness = st.selectbox("Completeness", COMPLETENESS_VALUES)
                terminal_sec = float(
                    st.number_input("Terminal time (s)", min_value=0.0, value=end_sec, step=0.04)
                )
                terminal_frame = int(
                    st.number_input("Terminal frame", min_value=0, value=end_frame)
                )
                terminal_cause = st.selectbox(
                    "Forced-out cause", ("",) + tuple(sorted(FORCED_OUT_CAUSES))
                )
                shot_body_part = st.selectbox(
                    "Shot body part",
                    ("", "right_foot", "left_foot", "head", "other", "unknown"),
                )
                shot_outcome = st.selectbox("Shot outcome", ("",) + SHOT_OUTCOMES)
                beneficiary_team_id = st.text_input("Beneficiary team ID", value="")
                last_touch_team_id = st.text_input("Last-touch team ID", value="")
                replay = st.checkbox("Replay/cutaway", value=False)
                confidence = float(st.slider("Annotation confidence", 0.0, 1.0, 0.9, 0.05))
                reason_codes_raw = st.text_input(
                    "Reason codes (comma-separated)", value="clear_terminal"
                )
                notes = st.text_area("Notes")
                annotator_id = st.text_input("Annotator ID", value="expert-1")
                annotation_round = int(
                    st.number_input("Annotation round", min_value=1, value=1)
                )
                submitted = st.form_submit_button("Save Attack-v2 annotation")

            if submitted:
                terminal_known = terminal_type != "unknown"
                item = GoldSequenceAnnotationV2(
                    match_id=match_id,
                    period=period,
                    stable_team_id=stable_team_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=label,
                    terminal_type=terminal_type,
                    terminal_frame=terminal_frame if terminal_known else None,
                    terminal_sec=terminal_sec if terminal_known else None,
                    terminal_cause=terminal_cause or None,
                    shot_body_part=shot_body_part or None,
                    shot_outcome=shot_outcome or None,
                    beneficiary_team_id=beneficiary_team_id or None,
                    last_touch_team_id=last_touch_team_id or None,
                    replay=replay,
                    completeness=completeness,
                    confidence=confidence,
                    reason_codes=tuple(
                        value.strip() for value in reason_codes_raw.split(",") if value.strip()
                    ),
                    notes=notes,
                    annotator_id=annotator_id,
                    annotated_at=utc_now_iso(),
                    annotation_round=annotation_round,
                ).with_stable_id()
                try:
                    item.validate()
                    if gold_split(match_id) == "test":
                        output_path = append_sealed_test_annotation_jsonl(
                            PROJECT_ROOT / "data" / "sealed-test", item
                        )
                    else:
                        output_path = (
                            PROJECT_ROOT
                            / "data"
                            / f"attack-v2-{gold_split(match_id)}.jsonl"
                        )
                        append_gold_annotation_jsonl(output_path, item)
                    terminal_note = (
                        "qualifying terminal"
                        if terminal_type in ATTACK_TERMINALS
                        else "non-attack terminal"
                    )
                    st.success(f"Saved {item.annotation_id} · {terminal_note} → {output_path}")
                except (OSError, PermissionError, ValueError) as error:
                    st.error(str(error))
        else:
            st.error(
                "Manifest này không dùng schema `attack-gold-v2.0`. Candidate/segment V1 "
                "không được mở trong workflow gán nhãn attack/effectiveness; hãy tạo continuous "
                "gold manifest bằng lệnh `build-gold-manifest`."
            )
