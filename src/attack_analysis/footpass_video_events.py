"""FOOTPASS TAAD action spotting on an uploaded broadcast clip.

FOOTPASS ships player tracklets from tactical HDF5. An uploaded clip has none,
so YOLO tracks stand in for them: each 50-frame window feeds up to 26 tracked
player boxes to the TAAD checkpoint, exactly as ``run_TAAD_on_matches.py``
feeds the 26 role slots. TAAD scores every slot independently, so the slot
order carries no role meaning here.

Highlight clips cut between the main camera, close-ups, celebrations and
replays. TAAD was trained on the main camera only, so it runs on wide-view
segments. A wide segment is a run of frames where many small players are
visible; close-ups and replays show a few large players.
"""

from __future__ import annotations

import importlib.util
import json
import os
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from .attack_v2 import AttackV2DecodeResult
from .models import PITCH_LENGTH, PITCH_WIDTH, Event, SourceTier
from .observations import FrameObservation
from .v2_adapters import decode_event_stream_v2

TAAD_CLASS_NAMES = (
    "background",
    "drive",
    "pass",
    "cross",
    "throw_in",
    "shot",
    "header",
    "tackle",
    "block",
)
TAAD_SLOTS = 26
TAAD_INPUT_SIZE = (640, 352)
_FOOTPASS_ROI_FRAME = (1920.0, 1080.0)
_ROI_EXPAND = 1.125
_ROI_DUMMY = (50.0, 50.0, 72.5, 99.0)
PLAYER_TYPES = frozenset({"player", "goalkeeper"})
TERMINAL_ACTIONS = frozenset({"shot", "header"})
ATTACKING_ACTIONS = frozenset({"drive", "pass", "cross", "shot", "header"})


@dataclass(frozen=True)
class BroadcastViewConfig:
    # Near goal the main camera often holds only 4-5 small players.
    min_players: int = 4
    max_player_height_ratio: float = 0.15
    smoothing_frames: int = 13
    min_segment_sec: float = 3.0

    def __post_init__(self) -> None:
        if self.min_players < 1:
            raise ValueError("min_players must be positive")
        if not 0.0 < self.max_player_height_ratio < 1.0:
            raise ValueError("max_player_height_ratio must be between 0 and 1")
        if self.smoothing_frames < 1 or self.min_segment_sec <= 0:
            raise ValueError("smoothing and minimum segment length must be positive")


@dataclass(frozen=True)
class WideSegment:
    start_index: int
    end_index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class TaadDetection:
    frame_index: int
    timestamp_sec: float
    action: str
    score: float
    track_id: str
    team: str | None
    pitch_x: float | None
    pitch_y: float | None
    segment_index: int


def _is_wide_frame(frame: FrameObservation, frame_height: float, config: BroadcastViewConfig) -> bool:
    heights = [
        item.bbox.y2 - item.bbox.y1
        for item in frame.objects
        if item.object_type in PLAYER_TYPES
    ]
    if len(heights) < config.min_players:
        return False
    return median(heights) <= config.max_player_height_ratio * frame_height


def wide_view_segments(
    frames: Sequence[FrameObservation],
    *,
    frame_height: float,
    config: BroadcastViewConfig | None = None,
) -> tuple[WideSegment, ...]:
    """Return main-camera runs; close-ups, celebrations and replays split them."""

    settings = config or BroadcastViewConfig()
    if frame_height <= 0:
        raise ValueError("frame_height must be positive")
    raw = [_is_wide_frame(frame, frame_height, settings) for frame in frames]
    half = settings.smoothing_frames // 2
    smoothed = []
    for index in range(len(raw)):
        window = raw[max(0, index - half) : index + half + 1]
        smoothed.append(sum(window) * 2 > len(window))
    for index, frame in enumerate(frames):
        if frame.scene_cut:
            smoothed[index] = False

    segments: list[WideSegment] = []
    start: int | None = None
    for index, wide in enumerate([*smoothed, False]):
        if wide and start is None:
            start = index
        elif not wide and start is not None:
            end = index - 1
            segment = WideSegment(
                start_index=start,
                end_index=end,
                start_sec=frames[start].timestamp_sec,
                end_sec=frames[end].timestamp_sec,
            )
            if segment.duration_sec >= settings.min_segment_sec:
                segments.append(segment)
            start = None
    return tuple(segments)


def _candidate_roots(start: Path) -> Iterable[Path]:
    configured = os.environ.get("FOOTPASS_ROOT", "").strip()
    if configured:
        yield Path(configured)
    for parent in (start, *start.parents):
        yield parent / "FOOTPASS"
    # OneDrive can store the same Vietnamese folder name in NFC and NFD forms;
    # only one of them holds the FOOTPASS checkout.
    for parent in start.parents:
        grand = parent.parent
        if not grand.is_dir():
            continue
        target = unicodedata.normalize("NFC", parent.name)
        for sibling in grand.iterdir():
            if sibling != parent and unicodedata.normalize("NFC", sibling.name) == target:
                relative = start.relative_to(parent)
                for step in (sibling / relative, *(sibling / relative).parents):
                    yield step / "FOOTPASS"


def resolve_footpass_root(start: Path | None = None) -> Path:
    origin = (start or Path(__file__)).resolve()
    for candidate in _candidate_roots(origin):
        if (candidate / "models" / "model_TAAD_baseline.py").is_file():
            return candidate
    raise FileNotFoundError(
        "FOOTPASS checkout with models/model_TAAD_baseline.py not found; set FOOTPASS_ROOT"
    )


def default_taad_checkpoint(footpass_root: Path) -> Path:
    return footpass_root / "runs" / "TAAD_Baseline_h100" / "checkpoints" / "best_model.pt"


def load_taad_model(footpass_root: Path, checkpoint: Path, device: str = "cuda:0") -> Any:
    import torch

    module_path = footpass_root / "models" / "model_TAAD_baseline.py"
    spec = importlib.util.spec_from_file_location("footpass_model_taad_baseline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.X3D_TAAD_Baseline()
    payload = torch.load(str(checkpoint), map_location=torch.device("cpu"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def _slot_tracks(window: Sequence[FrameObservation]) -> list[str]:
    presence: dict[str, int] = {}
    for frame in window:
        for item in frame.objects:
            if item.object_type in PLAYER_TYPES and item.track_id is not None:
                presence[item.track_id] = presence.get(item.track_id, 0) + 1
    ranked = sorted(presence, key=lambda track: (-presence[track], track))
    return ranked[:TAAD_SLOTS]


def _roi(item: Any, width: float, height: float) -> tuple[float, float, float, float]:
    sx = _FOOTPASS_ROI_FRAME[0] / width
    sy = _FOOTPASS_ROI_FRAME[1] / height
    x = item.bbox.x1 * sx
    y = item.bbox.y1 * sy
    w = (item.bbox.x2 - item.bbox.x1) * sx
    h = (item.bbox.y2 - item.bbox.y1) * sy
    tlx = max(min(1920, int(x - ((_ROI_EXPAND - 1.0) * w // 2))), 0)
    tly = max(min(1080, int(y - ((_ROI_EXPAND - 1.0) * h // 2))), 0)
    brx = max(min(1920, int(x + _ROI_EXPAND * w)), 0)
    bry = max(min(1080, int(y + _ROI_EXPAND * h)), 0)
    return (int(tlx / 3), int(tly / 3.068181), int(brx / 3), int(bry / 3.068181))


def _read_frames(video_path: Path, start: int, count: int) -> Any:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(count):
            ok, image = capture.read()
            if not ok:
                break
            image = cv2.resize(image, TAAD_INPUT_SIZE, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        return np.zeros((0, TAAD_INPUT_SIZE[1], TAAD_INPUT_SIZE[0], 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def _window_logits(
    model: Any,
    images: Any,
    window: Sequence[FrameObservation],
    *,
    width: float,
    height: float,
    device: str,
) -> tuple[Any, list[str]]:
    import numpy as np
    import torch

    clip_length = 50
    slots = _slot_tracks(window)
    length = len(window)
    padded = np.zeros((clip_length, TAAD_INPUT_SIZE[1], TAAD_INPUT_SIZE[0], 3), dtype=np.float32)
    padded[:length] = images[:length]
    rois = np.empty((TAAD_SLOTS, clip_length, 5), dtype=np.float64)
    masks = np.zeros((TAAD_SLOTS, clip_length), dtype=np.float64)
    for slot in range(TAAD_SLOTS):
        for t in range(clip_length):
            rois[slot, t, 0] = t
            rois[slot, t, 1:] = _ROI_DUMMY
    for t, frame in enumerate(window):
        boxes = {
            item.track_id: item
            for item in frame.objects
            if item.object_type in PLAYER_TYPES and item.track_id is not None
        }
        for slot, track_id in enumerate(slots):
            item = boxes.get(track_id)
            if item is None:
                continue
            rois[slot, t, 1:] = _roi(item, width, height)
            masks[slot, t] = 1.0
    clip = torch.from_numpy(padded).to(device)
    clip = ((clip / 255.0 - 0.45) / 0.225).half()
    clip = clip.reshape(1, clip_length, TAAD_INPUT_SIZE[1], TAAD_INPUT_SIZE[0], 3).permute(0, 4, 1, 2, 3)
    roi_tensor = torch.from_numpy(rois.reshape(TAAD_SLOTS, 1, clip_length, 5)).float().to(device).permute(1, 0, 2, 3)
    mask_tensor = torch.from_numpy(masks.reshape(TAAD_SLOTS, 1, clip_length)).float().to(device).permute(1, 0, 2)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model([clip, roi_tensor, mask_tensor])
    # (1, 9, M, T) -> (9, M, T); slots past len(slots) are masked, as in FOOTPASS.
    values = logits[0].float().cpu().numpy()[:, :, :length]
    return values, slots


def _softmax(values: Any) -> Any:
    import numpy as np

    shifted = values - values.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)


@dataclass
class SegmentTaadOutput:
    """TAAD output on one wide segment, indexed by segment-local frame.

    ``logits`` keeps the raw two-pass average that FOOTPASS DST consumes;
    ``probabilities`` is the softmax used for peak picking. Both are only
    filled on frames where the track is visible. ``empty_logits`` is the mean
    output on masked slots, which DST sees for players off camera.
    """

    probabilities: dict[str, Any]
    logits: dict[str, Any]
    present: dict[str, Any]
    empty_logits: Any


def _segment_track_outputs(
    model: Any,
    video_path: Path,
    frames: Sequence[FrameObservation],
    segment: WideSegment,
    *,
    width: float,
    height: float,
    device: str,
) -> SegmentTaadOutput:
    """Average two half-shifted 50-frame passes, like ``run_TAAD_on_matches``."""

    import numpy as np

    clip_length = 50
    first = segment.start_index
    last = segment.end_index
    total = last - first + 1
    images = _read_frames(video_path, frames[first].frame_index, total)
    total = min(total, len(images))
    prob_sums: dict[str, Any] = {}
    logit_sums: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    empty_sum = np.zeros(len(TAAD_CLASS_NAMES), dtype=np.float64)
    empty_count = 0
    for offset in (0, clip_length // 2):
        for local in range(offset, total, clip_length):
            window_end = min(local + clip_length, total)
            window = frames[first + local : first + window_end]
            if not window:
                continue
            logits, slots = _window_logits(
                model,
                images[local:window_end],
                window,
                width=width,
                height=height,
                device=device,
            )
            if len(slots) < TAAD_SLOTS:
                empty = logits[:, len(slots) :, :]
                empty_sum += empty.sum(axis=(1, 2))
                empty_count += empty.shape[1] * empty.shape[2]
            probabilities = _softmax(logits)
            for slot, track_id in enumerate(slots):
                if track_id not in prob_sums:
                    prob_sums[track_id] = np.zeros((len(TAAD_CLASS_NAMES), total), dtype=np.float64)
                    logit_sums[track_id] = np.zeros((len(TAAD_CLASS_NAMES), total), dtype=np.float64)
                    counts[track_id] = np.zeros(total, dtype=np.float64)
                present = np.array(
                    [
                        any(
                            item.track_id == track_id and item.object_type in PLAYER_TYPES
                            for item in frame.objects
                        )
                        for frame in window
                    ],
                    dtype=bool,
                )
                span = slice(local, window_end)
                width_now = window_end - local
                prob_sums[track_id][:, span][:, present] += probabilities[:, slot, :width_now][:, present]
                logit_sums[track_id][:, span][:, present] += logits[:, slot, :width_now][:, present]
                counts[track_id][span][present] += 1.0
                empty_sum += logits[:, slot, :width_now][:, ~present].sum(axis=1)
                empty_count += int((~present).sum())
    averaged: dict[str, Any] = {}
    averaged_logits: dict[str, Any] = {}
    present_masks: dict[str, Any] = {}
    for track_id, total_prob in prob_sums.items():
        count = counts[track_id]
        seen = count > 0
        result = np.zeros_like(total_prob)
        result[0, :] = 1.0
        result[:, seen] = total_prob[:, seen] / count[seen]
        averaged[track_id] = result
        raw = np.zeros_like(total_prob)
        raw[:, seen] = logit_sums[track_id][:, seen] / count[seen]
        averaged_logits[track_id] = raw
        present_masks[track_id] = seen
    empty_logits = (
        empty_sum / empty_count
        if empty_count
        else np.zeros(len(TAAD_CLASS_NAMES), dtype=np.float64)
    )
    return SegmentTaadOutput(
        probabilities=averaged,
        logits=averaged_logits,
        present=present_masks,
        empty_logits=empty_logits,
    )


def _peak_pick(probabilities: Any, nms_window: int, min_score: float) -> list[tuple[int, int, float]]:
    """FOOTPASS post-processing: argmax per frame, then temporal NMS per class."""

    import numpy as np

    classes = probabilities.argmax(axis=0)
    scores = probabilities.max(axis=0)
    picked: list[tuple[int, int, float]] = []
    for class_id in range(1, len(TAAD_CLASS_NAMES)):
        times = np.where(classes == class_id)[0]
        if times.size == 0:
            continue
        order = times[np.argsort(-scores[times])]
        suppressed: set[int] = set()
        for t in order:
            t = int(t)
            if t in suppressed:
                continue
            if scores[t] >= min_score:
                picked.append((t, class_id, float(scores[t])))
            for other in order:
                if abs(int(other) - t) <= nms_window:
                    suppressed.add(int(other))
    return picked


def run_taad(
    video_path: Path,
    frames: Sequence[FrameObservation],
    *,
    width: float,
    height: float,
    segments: Sequence[WideSegment],
    footpass_root: Path | None = None,
    checkpoint: Path | None = None,
    device: str = "cuda:0",
    nms_window: int = 25,
    min_score: float = 0.15,
) -> tuple[tuple[TaadDetection, ...], tuple[SegmentTaadOutput, ...]]:
    if any(
        later.frame_index - earlier.frame_index != 1 for earlier, later in zip(frames, frames[1:])
    ):
        raise ValueError("TAAD needs one observation per video frame; re-run vision with stride 1")
    root = footpass_root or resolve_footpass_root()
    model = load_taad_model(root, checkpoint or default_taad_checkpoint(root), device=device)
    outputs = tuple(
        _segment_track_outputs(
            model,
            video_path,
            frames,
            segment,
            width=width,
            height=height,
            device=device,
        )
        for segment in segments
    )
    return peak_pick_detections(
        outputs, segments, frames, nms_window=nms_window, min_score=min_score
    ), outputs


def detect_taad_actions(
    video_path: Path,
    frames: Sequence[FrameObservation],
    *,
    width: float,
    height: float,
    segments: Sequence[WideSegment],
    footpass_root: Path | None = None,
    checkpoint: Path | None = None,
    device: str = "cuda:0",
    nms_window: int = 25,
    min_score: float = 0.15,
) -> tuple[TaadDetection, ...]:
    detections, _outputs = run_taad(
        video_path,
        frames,
        width=width,
        height=height,
        segments=segments,
        footpass_root=footpass_root,
        checkpoint=checkpoint,
        device=device,
        nms_window=nms_window,
        min_score=min_score,
    )
    return detections


def peak_pick_detections(
    outputs: Sequence[SegmentTaadOutput],
    segments: Sequence[WideSegment],
    frames: Sequence[FrameObservation],
    *,
    nms_window: int = 25,
    min_score: float = 0.15,
) -> tuple[TaadDetection, ...]:
    detections: list[TaadDetection] = []
    for segment_index, (segment, output) in enumerate(zip(segments, outputs)):
        for track_id, probabilities in output.probabilities.items():
            for local, class_id, score in _peak_pick(probabilities, nms_window, min_score):
                frame = frames[segment.start_index + local]
                item = next(
                    (
                        obj
                        for obj in frame.objects
                        if obj.track_id == track_id and obj.object_type in PLAYER_TYPES
                    ),
                    None,
                )
                detections.append(
                    TaadDetection(
                        frame_index=frame.frame_index,
                        timestamp_sec=frame.timestamp_sec,
                        action=TAAD_CLASS_NAMES[class_id],
                        score=score,
                        track_id=track_id,
                        team=None if item is None else item.team,
                        pitch_x=None if item is None else item.pitch_x,
                        pitch_y=None if item is None else item.pitch_y,
                        segment_index=segment_index,
                    )
                )
    detections.sort(key=lambda item: (item.frame_index, -item.score))
    return tuple(detections)


def weak_shot_detections(
    outputs: Sequence[SegmentTaadOutput],
    segments: Sequence[WideSegment],
    frames: Sequence[FrameObservation],
    *,
    nms_window: int = 25,
    min_score: float = 0.15,
) -> tuple[TaadDetection, ...]:
    """Shot peaks that lose the per-frame argmax, usually to background.

    FOOTPASS peak picking never sees them. They are only a fallback finish
    (see :func:`highlight_attack_events`), never ordinary events.
    """

    import numpy as np

    shot = TAAD_CLASS_NAMES.index("shot")
    detections: list[TaadDetection] = []
    for segment_index, (segment, output) in enumerate(zip(segments, outputs)):
        for track_id, probabilities in output.probabilities.items():
            scores = np.where(np.asarray(output.present[track_id], bool), probabilities[shot], 0.0)
            times = np.where((scores >= min_score) & (probabilities.argmax(axis=0) != shot))[0]
            picked: list[int] = []
            for local in times[np.argsort(-scores[times])]:
                if any(abs(int(local) - other) <= nms_window for other in picked):
                    continue
                picked.append(int(local))
                frame = frames[segment.start_index + int(local)]
                item = next(
                    (
                        obj
                        for obj in frame.objects
                        if obj.track_id == track_id and obj.object_type in PLAYER_TYPES
                    ),
                    None,
                )
                detections.append(
                    TaadDetection(
                        frame_index=frame.frame_index,
                        timestamp_sec=frame.timestamp_sec,
                        action="shot",
                        score=float(scores[local]),
                        track_id=track_id,
                        team=None if item is None else item.team,
                        pitch_x=None if item is None else item.pitch_x,
                        pitch_y=None if item is None else item.pitch_y,
                        segment_index=segment_index,
                    )
                )
    detections.sort(key=lambda item: (item.frame_index, -item.score))
    return tuple(detections)


def _pitch_value(value: float | None, upper: float) -> float | None:
    if value is None or not 0.0 <= value <= upper:
        return None
    return float(value)


def _attacking_team(detections: Sequence[TaadDetection]) -> str | None:
    votes: dict[str, float] = {}
    for item in detections:
        if item.team and item.action in ATTACKING_ACTIONS:
            votes[item.team] = votes.get(item.team, 0.0) + item.score
    if not votes:
        return None
    return max(sorted(votes), key=lambda team: votes[team])


def _goalkeeper_share(frames: Sequence[FrameObservation], segment: WideSegment) -> dict[str, float]:
    seen: dict[str, int] = {}
    keeper: dict[str, int] = {}
    for frame in frames[segment.start_index : segment.end_index + 1]:
        for item in frame.objects:
            if item.object_type in PLAYER_TYPES and item.track_id is not None:
                seen[item.track_id] = seen.get(item.track_id, 0) + 1
                if item.object_type == "goalkeeper":
                    keeper[item.track_id] = keeper.get(item.track_id, 0) + 1
    return {track: keeper.get(track, 0) / count for track, count in seen.items()}


def _pick_terminal(
    candidates: Sequence[TaadDetection],
    *,
    team: str,
    goalkeeper_share: dict[str, float],
    window_sec: float,
    max_goalkeeper_share: float,
) -> TaadDetection:
    """Choose the finish among attempts that close the segment.

    TAAD fires shot/block on the goalkeeper who saves or collects the ball,
    a few frames after the real shot. Prefer an attacker of the attacking
    team, then a track YOLO rarely calls goalkeeper, then a real shot/header.
    """

    last = candidates[-1].timestamp_sec
    closing = [item for item in candidates if last - item.timestamp_sec <= window_sec]
    return max(
        closing,
        key=lambda item: (
            item.team == team,
            goalkeeper_share.get(item.track_id, 0.0) < max_goalkeeper_share,
            item.action in TERMINAL_ACTIONS,
            item.score,
            item.frame_index,
        ),
    )


def _defending_team(attacking: str, detections: Sequence[TaadDetection]) -> str:
    others = sorted({item.team for item in detections if item.team and item.team != attacking})
    return others[0] if others else f"not_{attacking}"


def _segment_event(
    item: TaadDetection,
    *,
    event_type: str,
    team: str,
    match_id: str,
    left_to_right: bool | None = None,
    source: str = "footpass_taad",
) -> Event:
    """`left_to_right` is the actor team's raw direction; x is mirrored like FOOTPASS."""

    x = _pitch_value(item.pitch_x, PITCH_LENGTH)
    if x is not None and left_to_right is False:
        x = PITCH_LENGTH - x
    return Event(
        timestamp=item.timestamp_sec,
        event_type=event_type,
        team=team,
        x=x,
        y=_pitch_value(item.pitch_y, PITCH_WIDTH),
        frame=item.frame_index,
        player_id=item.track_id,
        confidence=min(1.0, max(0.0, item.score)),
        source=source,
        evidence_id=f"{source}:{match_id}:{item.frame_index}:{item.track_id}:{item.action}",
    )


@dataclass(frozen=True)
class HighlightAttack:
    """One live wide-view segment read as one attack.

    ``terminal`` is the last TAAD shot or header in the segment. A highlight
    edit cuts away right after the chance, so the last attempt is the one the
    editor kept. The attacking team is a score-weighted vote over ball actions
    because per-frame YOLO jersey colours flip. Every event's team is that
    segment-level vote, not the actor's own jersey.

    `left_to_right` is the attacking team's raw direction on the calibrated
    pitch (`True`: towards x=105), read from where the finish happened.
    Event coordinates are mirrored per actor team, as FOOTPASS stores them.
    """

    segment: WideSegment
    segment_index: int
    team: str | None
    events: tuple[Event, ...]
    terminal: TaadDetection | None
    reason: str
    left_to_right: bool | None = None
    defending_team: str | None = None

    def team_directions(self) -> dict[str, bool]:
        if self.team is None or self.left_to_right is None:
            return {}
        directions = {self.team: self.left_to_right}
        if self.defending_team:
            directions[self.defending_team] = not self.left_to_right
        return directions


def _attack_left_to_right(
    terminal: TaadDetection,
    detections: Sequence[TaadDetection],
    team: str,
) -> bool | None:
    if terminal.pitch_x is not None:
        return terminal.pitch_x > PITCH_LENGTH / 2.0
    xs = sorted(
        item.pitch_x
        for item in detections
        if item.team == team and item.action in ATTACKING_ACTIONS and item.pitch_x is not None
    )
    if not xs:
        return None
    return xs[len(xs) // 2] > PITCH_LENGTH / 2.0


def _last_ball_carrier(
    weak: TaadDetection,
    frames: Sequence[FrameObservation],
    segment: WideSegment,
    *,
    window_sec: float = 1.0,
    min_ball_confidence: float = 0.25,
    max_feet_distance: float = 1.0,
) -> TaadDetection | None:
    """The player last seen with the ball before a weak shot peak.

    A weak TAAD peak often lands on whoever is nearest when the ball flies by,
    such as a beaten goalkeeper. The striker is the last player whose feet sat
    within ``max_feet_distance`` body heights of a confident ball.
    """

    for frame in reversed(frames[segment.start_index : segment.end_index + 1]):
        if frame.timestamp_sec > weak.timestamp_sec:
            continue
        if frame.timestamp_sec < weak.timestamp_sec - window_sec:
            break
        best: tuple[float, Any] | None = None
        for ball in frame.objects:
            if ball.object_type != "ball" or ball.confidence < min_ball_confidence:
                continue
            ball_x = (ball.bbox.x1 + ball.bbox.x2) / 2.0
            ball_y = (ball.bbox.y1 + ball.bbox.y2) / 2.0
            for item in frame.objects:
                if item.object_type not in PLAYER_TYPES or item.track_id is None:
                    continue
                height = max(1.0, item.bbox.y2 - item.bbox.y1)
                distance = (
                    ((item.bbox.x1 + item.bbox.x2) / 2.0 - ball_x) ** 2
                    + (item.bbox.y2 - ball_y) ** 2
                ) ** 0.5 / height
                if distance <= max_feet_distance and (best is None or distance < best[0]):
                    best = (distance, item)
        if best is not None:
            item = best[1]
            return TaadDetection(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                action="shot",
                score=weak.score,
                track_id=str(item.track_id),
                team=item.team,
                pitch_x=item.pitch_x,
                pitch_y=item.pitch_y,
                segment_index=weak.segment_index,
            )
    return None


def _attack_view_x(item: TaadDetection, left_to_right: bool | None) -> float | None:
    """Distance-to-goal frame: ``None`` direction means the goal nearest the player."""

    x = _pitch_value(item.pitch_x, PITCH_LENGTH)
    if x is None:
        return None
    if left_to_right is None:
        return max(x, PITCH_LENGTH - x)
    return x if left_to_right else PITCH_LENGTH - x


def highlight_attack_events(
    detections: Sequence[TaadDetection],
    segments: Sequence[WideSegment],
    frames: Sequence[FrameObservation],
    *,
    match_id: str,
    min_terminal_score: float = 0.15,
    finish_window_sec: float = 1.0,
    max_goalkeeper_share: float = 0.05,
    source: str = "footpass_taad",
    weak_shots: Sequence[TaadDetection] = (),
    min_finish_x: float = 75.0,
    weak_shot_min_x: float = 83.0,
    weak_shot_max_offset_y: float = 20.0,
) -> tuple[HighlightAttack, ...]:
    """``weak_shots`` (from :func:`weak_shot_detections`) replace a terminal that
    sits more than ``105 - min_finish_x`` m from goal: a midfield header is not an
    attempt, and the real finish after it often loses the argmax to background.
    Only the last weak shot near goal (attack-view x >= ``weak_shot_min_x``),
    not by the defending team or a goalkeeper track, is used.
    """

    results: list[HighlightAttack] = []
    for segment_index, segment in enumerate(segments):
        inside = sorted(
            (item for item in detections if item.segment_index == segment_index),
            key=lambda item: (item.frame_index, -item.score),
        )
        team = _attacking_team(inside)
        if team is None:
            results.append(HighlightAttack(segment, segment_index, None, (), None, "no_team"))
            continue
        # FOOTPASS "block" is a defender's action. Fired on an attacking player
        # it is almost always a shot under pressure (the clip's empty-net finish).
        terminals = [
            item
            for item in inside
            if item.score >= min_terminal_score
            and (item.action in TERMINAL_ACTIONS or (item.action == "block" and item.team == team))
        ]
        if not terminals:
            results.append(HighlightAttack(segment, segment_index, team, (), None, "no_shot_or_header"))
            continue
        goalkeeper_share = _goalkeeper_share(frames, segment)
        terminal = _pick_terminal(
            terminals,
            team=team,
            goalkeeper_share=goalkeeper_share,
            window_sec=finish_window_sec,
            max_goalkeeper_share=max_goalkeeper_share,
        )
        defending = _defending_team(team, inside)
        left_to_right = _attack_left_to_right(terminal, inside, team)
        reason = "terminal_found"
        terminal_x = _attack_view_x(terminal, left_to_right)
        if terminal_x is not None and terminal_x < min_finish_x:
            later = [
                item
                for item in weak_shots
                if item.segment_index == segment_index
                and item.frame_index > terminal.frame_index
                and item.team in {team, None}
                and goalkeeper_share.get(item.track_id, 0.0) < max_goalkeeper_share
                and item.pitch_y is not None
                and abs(item.pitch_y - PITCH_WIDTH / 2.0) <= weak_shot_max_offset_y
                and (_attack_view_x(item, None) or 0.0) >= weak_shot_min_x
            ]
            if later:
                weak = max(later, key=lambda item: item.frame_index)
                carrier = _last_ball_carrier(weak, frames, segment)
                terminal = (
                    carrier
                    if carrier is not None
                    and carrier.pitch_x is not None
                    and carrier.frame_index > terminal.frame_index
                    else weak
                )
                left_to_right = _attack_left_to_right(terminal, inside, team)
                reason = "weak_finish_after_far_terminal"
        defending_direction = None if left_to_right is None else not left_to_right
        start_frame = frames[segment.start_index]
        events = [
            Event(
                timestamp=start_frame.timestamp_sec,
                event_type="possession_start",
                team=team,
                frame=start_frame.frame_index,
                source="footpass_wide_segment",
                evidence_id=f"segment:{match_id}:{start_frame.frame_index}",
            )
        ]
        for item in inside:
            if item is terminal:
                continue
            if item.frame_index > terminal.frame_index:
                break
            if item.action in {"drive", "pass", "cross"}:
                events.append(
                    _segment_event(
                        item,
                        event_type=item.action,
                        team=team,
                        match_id=match_id,
                        left_to_right=left_to_right,
                        source=source,
                    )
                )
            elif item.action in TERMINAL_ACTIONS:
                event_type = "shot_candidate" if item.action == "shot" else "header"
                events.append(
                    _segment_event(
                        item,
                        event_type=event_type,
                        team=team,
                        match_id=match_id,
                        left_to_right=left_to_right,
                        source=source,
                    )
                )
            elif item.action in {"tackle", "block"}:
                events.append(
                    _segment_event(
                        item,
                        event_type=item.action,
                        team=defending,
                        match_id=match_id,
                        left_to_right=defending_direction,
                        source=source,
                    )
                )
        events.append(
            _segment_event(
                terminal,
                event_type="headed_shot" if terminal.action == "header" else "shot",
                team=team,
                match_id=match_id,
                left_to_right=left_to_right,
                source=source,
            )
        )
        results.append(
            HighlightAttack(
                segment,
                segment_index,
                team,
                tuple(events),
                terminal,
                reason,
                left_to_right=left_to_right,
                defending_team=defending,
            )
        )
    return tuple(results)


def decode_highlight_attacks(
    highlights: Sequence[HighlightAttack],
    *,
    match_id: str,
    fps: float,
) -> AttackV2DecodeResult:
    """Decode each segment alone: a cut ends the footage, not the attack."""

    possessions = []
    attacks = []
    unassigned: list[str] = []
    for highlight in highlights:
        if not highlight.events:
            continue
        _actions, decoded = decode_event_stream_v2(
            highlight.events,
            match_id=match_id,
            fps=fps,
            source_tier=SourceTier.VIDEO_ONLY,
            replay_status="live",
            team_left_to_right=highlight.team_directions(),
        )
        possessions.extend(decoded.possessions)
        attacks.extend(decoded.attacks)
        unassigned.extend(decoded.unassigned_action_ids)
    return AttackV2DecodeResult(
        possessions=tuple(possessions),
        attacks=tuple(attacks),
        unassigned_action_ids=tuple(sorted(set(unassigned))),
    )


def save_taad_outputs(path: Path, outputs: Sequence[SegmentTaadOutput]) -> None:
    import numpy as np

    arrays: dict[str, Any] = {}
    for index, output in enumerate(outputs):
        arrays[f"s{index}__empty"] = np.asarray(output.empty_logits, dtype=np.float32)
        for track_id, logits in output.logits.items():
            arrays[f"s{index}__logits__{track_id}"] = logits.astype(np.float32)
            arrays[f"s{index}__probs__{track_id}"] = output.probabilities[track_id].astype(np.float32)
            arrays[f"s{index}__present__{track_id}"] = output.present[track_id]
    partial = path.with_name(path.stem + ".partial.npz")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(path)


def load_taad_outputs(path: Path, segment_count: int) -> tuple[SegmentTaadOutput, ...]:
    import numpy as np

    outputs = [
        SegmentTaadOutput(probabilities={}, logits={}, present={}, empty_logits=None)
        for _ in range(segment_count)
    ]
    with np.load(path) as payload:
        for key in payload.files:
            index_text, kind, *rest = key.split("__")
            index = int(index_text[1:])
            if index >= segment_count:
                raise ValueError("TAAD logits cache does not match the segments")
            output = outputs[index]
            if kind == "empty":
                output.empty_logits = payload[key].astype(np.float64)
                continue
            track_id = "__".join(rest)
            values = payload[key]
            if kind == "logits":
                output.logits[track_id] = values.astype(np.float64)
            elif kind == "probs":
                output.probabilities[track_id] = values.astype(np.float64)
            elif kind == "present":
                output.present[track_id] = values.astype(bool)
    if any(output.empty_logits is None for output in outputs):
        raise ValueError("TAAD logits cache is incomplete")
    return tuple(outputs)


def detections_to_records(detections: Iterable[TaadDetection]) -> list[dict[str, object]]:
    return [asdict(item) for item in detections]


def detections_from_records(records: Iterable[dict[str, Any]]) -> tuple[TaadDetection, ...]:
    return tuple(TaadDetection(**record) for record in records)


def extract_highlight_attacks(
    video_path: Path,
    frames: Sequence[FrameObservation],
    *,
    width: float,
    height: float,
    fps: float,
    match_id: str,
    device: str = "cuda:0",
    cache_path: Path | None = None,
) -> tuple[tuple[WideSegment, ...], tuple[TaadDetection, ...], tuple[HighlightAttack, ...], AttackV2DecodeResult]:
    """Wide-view segments, TAAD actions, one attack per live segment."""

    result = extract_taad_highlights(
        video_path,
        frames,
        width=width,
        height=height,
        fps=fps,
        match_id=match_id,
        device=device,
        cache_path=cache_path,
    )
    return result.segments, result.detections, result.highlights, result.decoded


@dataclass(frozen=True)
class TaadHighlightResult:
    segments: tuple[WideSegment, ...]
    detections: tuple[TaadDetection, ...]
    highlights: tuple[HighlightAttack, ...]
    decoded: AttackV2DecodeResult
    outputs: tuple[SegmentTaadOutput, ...]


def extract_taad_highlights(
    video_path: Path,
    frames: Sequence[FrameObservation],
    *,
    width: float,
    height: float,
    fps: float,
    match_id: str,
    device: str = "cuda:0",
    cache_path: Path | None = None,
) -> TaadHighlightResult:
    """Like :func:`extract_highlight_attacks`, keeping TAAD logits for DST.

    ``cache_path`` is a JSON file naming the segments; the logits sit next to
    it as ``.npz``. Detections are re-peak-picked from cached probabilities so
    their pitch positions always follow the current observations.
    """

    segments = wide_view_segments(frames, frame_height=height)
    outputs: tuple[SegmentTaadOutput, ...] | None = None
    logits_path = cache_path.with_suffix(".npz") if cache_path is not None else None
    if (
        cache_path is not None
        and logits_path is not None
        and cache_path.is_file()
        and logits_path.is_file()
    ):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("segments") == [asdict(item) for item in segments]:
            try:
                outputs = load_taad_outputs(logits_path, len(segments))
            except (ValueError, OSError, KeyError):
                outputs = None
    if outputs is None:
        _detections, outputs = run_taad(
            video_path,
            frames,
            width=width,
            height=height,
            segments=segments,
            device=device,
        )
        if cache_path is not None and logits_path is not None:
            save_taad_outputs(logits_path, outputs)
            cache_path.write_text(
                json.dumps({"segments": [asdict(item) for item in segments]}),
                encoding="utf-8",
            )
    detections = peak_pick_detections(outputs, segments, frames)
    highlights = highlight_attack_events(
        detections,
        segments,
        frames,
        match_id=match_id,
        weak_shots=weak_shot_detections(outputs, segments, frames),
    )
    decoded = decode_highlight_attacks(highlights, match_id=match_id, fps=fps)
    return TaadHighlightResult(
        segments=segments,
        detections=detections,
        highlights=highlights,
        decoded=decoded,
        outputs=outputs,
    )


__all__ = [
    "ATTACKING_ACTIONS",
    "BroadcastViewConfig",
    "HighlightAttack",
    "SegmentTaadOutput",
    "TAAD_CLASS_NAMES",
    "TERMINAL_ACTIONS",
    "TaadDetection",
    "TaadHighlightResult",
    "WideSegment",
    "decode_highlight_attacks",
    "default_taad_checkpoint",
    "detect_taad_actions",
    "detections_from_records",
    "detections_to_records",
    "extract_highlight_attacks",
    "extract_taad_highlights",
    "highlight_attack_events",
    "load_taad_model",
    "load_taad_outputs",
    "peak_pick_detections",
    "weak_shot_detections",
    "resolve_footpass_root",
    "run_taad",
    "save_taad_outputs",
    "wide_view_segments",
]
