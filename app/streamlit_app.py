from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from attack_analysis.agent_tools import EvidenceToolRegistry
from attack_analysis.annotations import AttackAnnotation
from attack_analysis.attack_features import build_attack_features
from attack_analysis.benchmarks import FrozenBenchmarkStore
from attack_analysis.env import load_project_environment, openai_api_key_configured
from attack_analysis.footpass import FootpassSegmentationConfig, load_footpass_events, segment_footpass_attacks
from attack_analysis.llm import (
    DeterministicEvidenceClient,
    OpenAIResponsesClient,
    describe_openai_error,
)
from attack_analysis.models import AttackSequence
from attack_analysis.research import EvidenceBundle, EvaluationVerdict, insufficient_evidence_verdict
from attack_analysis.video import inspect_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_FOOTPASS = WORKSPACE_ROOT / "FOOTPASS"
DEFAULT_VIDEO_ROOT = Path(os.environ.get("FOOTPASS_VIDEOS_DIR", r"D:\videos"))
load_project_environment(PROJECT_ROOT)
LOGGER = logging.getLogger(__name__)


st.set_page_config(page_title="Football Attack Thesis", layout="wide")
st.title("Football attack effectiveness")
st.caption("Oracle versus predicted evidence, quality-gated verdicts and traceable claims")


@st.cache_data(show_spinner=False)
def playbyplay_keys(path: str) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(str(item) for item in payload.get("keys", payload.get("events", {}).keys()))


@st.cache_data(show_spinner="Loading FOOTPASS events and attacks...")
def load_attacks(
    playbyplay: str,
    match_key: str,
    source: str,
    tactical_h5: str | None,
    max_gap: float,
    min_confidence: float,
    model_name: str | None,
) -> tuple[dict[str, object], ...]:
    events = load_footpass_events(
        Path(playbyplay),
        match_key=match_key,
        source=source,
        tactical_h5=Path(tactical_h5) if tactical_h5 else None,
        min_confidence=min_confidence,
        model_name=model_name,
    )
    attacks = segment_footpass_attacks(
        events,
        match_id=match_key,
        config=FootpassSegmentationConfig(max_gap_sec=max_gap),
    )
    return tuple(item.to_dict() for item in attacks)


def build_bundle(sequence: AttackSequence, *, tactical_ready: bool) -> EvidenceBundle:
    features = build_attack_features(sequence)
    event_ready = len(sequence.events) > 0
    quality_gates = {
        "events": event_ready,
        "tracking": tactical_ready,
        "calibration": tactical_ready and any(item.x is not None for item in sequence.events),
    }
    limitations = tuple(
        f"Quality gate failed: {name}."
        for name, passed in quality_gates.items()
        if not passed
    )
    return EvidenceBundle(
        attack=features,
        quality_gates=quality_gates,
        quality_metrics={
            "event_confidence": round(sum(item.confidence for item in sequence.events) / len(sequence.events), 4),
        },
        source_ids=(sequence.source,),
        limitations=limitations,
    )


def render_attack(sequence: AttackSequence, bundle: EvidenceBundle, video_path: Path | None) -> None:
    left, middle, right = st.columns(3)
    left.metric("Events", bundle.attack.event_count)
    middle.metric("Progression", f"{bundle.attack.progression_m:.1f} m")
    right.metric("xG", f"{bundle.attack.xg_sum:.3f}")

    gate_columns = st.columns(3)
    for column, (name, passed) in zip(gate_columns, bundle.quality_gates.items()):
        column.metric(name.title(), "PASS" if passed else "FAIL")

    if video_path and video_path.exists():
        st.video(str(video_path), start_time=max(0, int(sequence.start_sec)))

    event_rows = [item.to_dict() for item in sequence.events]
    st.subheader("Event timeline")
    st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)

    points = pd.DataFrame(
        [{"x": item.x, "y": item.y, "action": item.event_type} for item in sequence.events if item.x is not None]
    )
    if not points.empty:
        st.subheader("Pitch coordinates (attacking direction left to right)")
        st.scatter_chart(points, x="x", y="y", color="action", height=320)

    with st.expander("Dữ liệu đầu vào có cấu trúc (JSON, dành cho nghiên cứu)"):
        st.json(bundle.to_dict(), expanded=False)


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
    "header": "Đánh đầu",
    "shot": "Dứt điểm",
    "goal": "Bàn thắng",
    "turnover": "Mất bóng",
}
SOURCE_LABELS = {
    "footpass_oracle": "FOOTPASS · Oracle",
    "footpass_predicted": "FOOTPASS · Predicted",
}


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


analysis_tab, upload_tab, annotation_tab = st.tabs(
    ["FOOTPASS analysis", "Exploratory upload", "Annotation"]
)

with analysis_tab:
    mode = st.radio("Evidence mode", ("oracle", "predicted"), horizontal=True)
    predicted_model = None
    min_confidence = 0.0
    if mode == "predicted":
        predicted_model = st.selectbox(
            "Action model",
            ("DST", "TAAD"),
            help="DST denoises the raw TAAD sequence and is the recommended predicted baseline.",
        )
        min_confidence = st.slider(
            "Action confidence threshold",
            0.0,
            1.0,
            0.15,
            0.05,
            help="0.15 is the official FOOTPASS evaluation threshold; use 0.60 for a precision-oriented ablation.",
        )
        if predicted_model == "DST":
            st.caption(
                "Validation: micro-F1 0.719, 1,215 attacks and 66 shot attacks at threshold 0.15."
            )
        else:
            st.caption(
                "Ablation only: raw TAAD micro-F1 0.410 and severe attack over-segmentation."
            )
    default_playbyplay = (
        DEFAULT_FOOTPASS / "playbyplay_GT" / "playbyplay_val.json"
        if mode == "oracle"
        else DEFAULT_FOOTPASS
        / "playbyplay_PRED"
        / f"playbyplay_{predicted_model}_val.json"
    )
    playbyplay = Path(
        st.text_input(
            "Play-by-play JSON",
            str(default_playbyplay),
            key=f"playbyplay_{mode}_{predicted_model or 'oracle'}",
        )
    )
    tactical = Path(
        st.text_input("Tactical HDF5", str(DEFAULT_FOOTPASS / "data" / "val_tactical_data.h5"))
    )
    max_gap = st.slider("Inter-action gap (seconds)", 2.0, 5.0, 4.0, 1.0)

    if playbyplay.exists():
        keys = playbyplay_keys(str(playbyplay))
        match_key = st.selectbox("Match half", keys)
        attacks_data = load_attacks(
            str(playbyplay),
            match_key,
            mode,
            str(tactical) if tactical.exists() else None,
            max_gap,
            min_confidence,
            predicted_model,
        )
        if attacks_data:
            labels = [
                f"{item['attack_id']} · {item['start_sec']:.1f}-{item['end_sec']:.1f}s · {len(item['events'])} events"
                for item in attacks_data
            ]
            selected = st.selectbox("Attack", range(len(labels)), format_func=lambda index: labels[index])
            sequence = AttackSequence.from_dict(attacks_data[selected])
            bundle = build_bundle(sequence, tactical_ready=tactical.exists())
            game_id = match_key.split("_H", 1)[0]
            video_path = DEFAULT_VIDEO_ROOT / f"{game_id}.mp4"
            render_attack(sequence, bundle, video_path)

            st.subheader("Đánh giá hiệu quả đợt tấn công")
            use_api = st.toggle("Sử dụng OpenAI Responses API", value=False)
            api_model = None
            if use_api:
                api_key_ready = openai_api_key_configured()
                st.caption(
                    "OPENAI_API_KEY đã được cấu hình."
                    if api_key_ready
                    else "Tiến trình Streamlit chưa nhận được OPENAI_API_KEY."
                )
                api_model = st.text_input(
                    "Model OpenAI",
                    value=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
                    help="Giữ cố định model này để thí nghiệm luận văn có thể tái lập.",
                ).strip()
            if st.button("Đánh giá đợt tấn công đã chọn", type="primary"):
                store = FrozenBenchmarkStore.from_jsonl(
                    PROJECT_ROOT / "data" / "benchmarks.jsonl",
                    PROJECT_ROOT / "data" / "identities.jsonl",
                )
                registry = EvidenceToolRegistry({sequence.attack_id: bundle}, store)
                try:
                    if use_api:
                        if not api_key_ready:
                            raise ValueError(
                                "OPENAI_API_KEY is missing. Add it to the project .env file and restart Streamlit."
                            )
                        if not api_model:
                            raise ValueError("OpenAI model snapshot is required")
                        client = OpenAIResponsesClient(
                            model=api_model,
                            project_root=PROJECT_ROOT,
                            response_language="Vietnamese",
                        )
                    else:
                        client = DeterministicEvidenceClient()
                    with st.spinner("Đang phân tích bằng chứng và kiểm chứng các nhận định..."):
                        verdict = client.evaluate(bundle, registry)
                    render_verdict(verdict, bundle, sequence)
                except Exception as error:
                    if use_api:
                        LOGGER.exception("OpenAI attack evaluation failed")
                        st.error(describe_openai_error(error))
                        st.warning(
                            "OpenAI verdict unavailable; showing the deterministic baseline so the demo remains usable."
                        )
                        fallback = DeterministicEvidenceClient().evaluate(bundle, registry)
                        render_verdict(fallback, bundle, sequence)
                    else:
                        st.error(str(error))
    else:
        st.warning("Play-by-play file was not found.")

with upload_tab:
    uploaded = st.file_uploader("Upload a broadcast clip", type=("mp4", "mkv", "mov"))
    if uploaded is not None:
        content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        cache_dir = PROJECT_ROOT / "runs" / "upload-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip_path = cache_dir / f"{digest}{Path(uploaded.name).suffix.lower()}"
        if not clip_path.exists():
            clip_path.write_bytes(content)
        st.video(str(clip_path))
        try:
            st.json(inspect_video(clip_path).to_dict())
        except Exception as error:
            st.error(str(error))
        verdict = insufficient_evidence_verdict(
            digest[:16],
            (
                "Uploaded clips are exploratory.",
                "No validated dynamic calibration is available.",
                "No quality-gated player-centric event stream is available.",
            ),
        )
        st.json(verdict.to_dict())

with annotation_tab:
    manifest_file = st.file_uploader("Annotation manifest JSON", type="json", key="annotation_manifest")
    if manifest_file is not None:
        manifest = json.loads(manifest_file.getvalue())
        candidates = manifest.get("attacks", ())
        if candidates:
            index = st.selectbox(
                "Sequence",
                range(len(candidates)),
                format_func=lambda value: candidates[value]["sequence"]["attack_id"],
            )
            sequence = AttackSequence.from_dict(candidates[index]["sequence"])
            bundle = build_bundle(sequence, tactical_ready=True)
            render_attack(sequence, bundle, None)
            label = st.selectbox("Label", ("ineffective", "promising", "effective"))
            score = st.slider("Score", 1, 5, 3)
            confidence = st.slider("Confidence", 0.0, 1.0, 0.8, 0.05)
            reason = st.text_area("Reason")
            output_path = Path(st.text_input("Annotation JSONL output", str(PROJECT_ROOT / "data" / "annotations.jsonl")))
            if st.button("Save annotation"):
                annotation = AttackAnnotation(
                    attack_id=sequence.attack_id,
                    match_id=sequence.match_id or "unknown",
                    label=label,
                    score=score,
                    confidence=confidence,
                    reason=reason,
                    annotated_at=datetime.now(timezone.utc).isoformat(),
                )
                try:
                    payload = json.dumps(annotation.to_dict(), ensure_ascii=False)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_path.open("a", encoding="utf-8") as handle:
                        handle.write(payload + "\n")
                    st.success(f"Saved {sequence.attack_id}")
                except ValueError as error:
                    st.error(str(error))
