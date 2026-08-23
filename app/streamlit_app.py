from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from attack_analysis.agent_tools import EvidenceToolRegistry
from attack_analysis.annotations import AttackAnnotation
from attack_analysis.attack_features import build_attack_features
from attack_analysis.benchmarks import FrozenBenchmarkStore
from attack_analysis.footpass import FootpassSegmentationConfig, load_footpass_events, segment_footpass_attacks
from attack_analysis.llm import DeterministicEvidenceClient, OpenAIResponsesClient
from attack_analysis.models import AttackSequence
from attack_analysis.research import EvidenceBundle, insufficient_evidence_verdict
from attack_analysis.video import inspect_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_FOOTPASS = WORKSPACE_ROOT / "FOOTPASS"
DEFAULT_VIDEO_ROOT = Path(os.environ.get("FOOTPASS_VIDEOS_DIR", r"D:\videos"))


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
) -> tuple[dict[str, object], ...]:
    events = load_footpass_events(
        Path(playbyplay),
        match_key=match_key,
        source=source,
        tactical_h5=Path(tactical_h5) if tactical_h5 else None,
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
    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)

    points = pd.DataFrame(
        [{"x": item.x, "y": item.y, "action": item.event_type} for item in sequence.events if item.x is not None]
    )
    if not points.empty:
        st.subheader("Pitch coordinates (attacking direction left to right)")
        st.scatter_chart(points, x="x", y="y", color="action", height=320)

    st.subheader("Structured evidence")
    st.json(bundle.to_dict(), expanded=False)


analysis_tab, upload_tab, annotation_tab = st.tabs(
    ["FOOTPASS analysis", "Exploratory upload", "Annotation"]
)

with analysis_tab:
    mode = st.radio("Evidence mode", ("oracle", "predicted"), horizontal=True)
    default_playbyplay = (
        DEFAULT_FOOTPASS / "playbyplay_GT" / "playbyplay_val.json"
        if mode == "oracle"
        else DEFAULT_FOOTPASS / "playbyplay_PRED" / "playbyplay_TAAD_val.json"
    )
    playbyplay = Path(st.text_input("Play-by-play JSON", str(default_playbyplay)))
    tactical = Path(
        st.text_input("Tactical HDF5", str(DEFAULT_FOOTPASS / "data" / "val_tactical_data.h5"))
    )
    max_gap = st.slider("Inter-action gap (seconds)", 2.0, 5.0, 4.0, 1.0)

    if playbyplay.exists():
        keys = playbyplay_keys(str(playbyplay))
        match_key = st.selectbox("Match half", keys)
        attacks_data = load_attacks(
            str(playbyplay), match_key, mode, str(tactical) if tactical.exists() else None, max_gap
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

            st.subheader("Grounded verdict")
            use_api = st.toggle("Use OpenAI Responses API", value=False)
            api_model = None
            if use_api:
                api_model = st.text_input(
                    "OpenAI model snapshot",
                    value=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
                    help="Pin this value for reproducible thesis experiments.",
                ).strip()
            if st.button("Evaluate selected attack", type="primary"):
                store = FrozenBenchmarkStore.from_jsonl(
                    PROJECT_ROOT / "data" / "benchmarks.jsonl",
                    PROJECT_ROOT / "data" / "identities.jsonl",
                )
                registry = EvidenceToolRegistry({sequence.attack_id: bundle}, store)
                try:
                    if use_api:
                        if not api_model:
                            raise ValueError("OpenAI model snapshot is required")
                        client = OpenAIResponsesClient(model=api_model, project_root=PROJECT_ROOT)
                    else:
                        client = DeterministicEvidenceClient()
                    verdict = client.evaluate(bundle, registry)
                    st.json(verdict.to_dict())
                except Exception as error:
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
