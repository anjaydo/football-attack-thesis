"""FOOTPASS DST on an uploaded broadcast clip (research only).

``run_DST_on_matches.py`` feeds DST two things per frame: TAAD logits for 26
role slots and a team vector (x, y, vx, vy, observed) for the same slots, where
slot = left_to_right * 13 + role_id - 1. FOOTPASS gets both from tactical
tracking data. Here the positions come from YOLO tracks projected with the
landmark calibration, and each track is given a role slot by matching its mean
position to the average FOOTPASS role position. Players off camera stay at the
FOOTPASS missing value (-15) for every channel. DST never saw that in
training, where all 22 players have positions, so its output is weaker here
than on FOOTPASS validation.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attack_v2 import AttackV2DecodeResult
from .footpass_video_events import (
    PLAYER_TYPES,
    TAAD_CLASS_NAMES,
    HighlightAttack,
    SegmentTaadOutput,
    TaadDetection,
    WideSegment,
    _goalkeeper_share,
    decode_highlight_attacks,
    highlight_attack_events,
    resolve_footpass_root,
)
from .models import PITCH_LENGTH, PITCH_WIDTH
from .observations import FrameObservation

DST_FRAMESPAN = 750
DST_MAX_EVENTS = 25
DST_INPUT_DIM = 364
DST_OUTPUT_DIM = 37
ROLES_PER_TEAM = 13
DST_SLOTS = 2 * ROLES_PER_TEAM
MISSING_VALUE = -15.0
# FOOTPASS speed = 223 x normalised displacement per frame (measured on val
# tactical data, game_18_H1).
FOOTPASS_VELOCITY_SCALE = 223.0
GOALKEEPER_ROLE = 1
# Mean normalised position per FOOTPASS role for a team attacking +x, from the
# val tactical data (games 18, 24 and 47). Roles 4 and 8 never occur there.
ROLE_TEMPLATE: Mapping[int, tuple[float, float]] = {
    1: (0.125, 0.507),
    2: (0.417, 0.249),
    3: (0.360, 0.467),
    5: (0.358, 0.553),
    6: (0.487, 0.479),
    7: (0.489, 0.577),
    9: (0.452, 0.551),
    10: (0.541, 0.308),
    11: (0.557, 0.708),
    12: (0.591, 0.479),
    13: (0.425, 0.771),
}
OUTFIELD_ROLES = tuple(role for role in ROLE_TEMPLATE if role != GOALKEEPER_ROLE)


def default_dst_checkpoint(footpass_root: Path) -> Path:
    return footpass_root / "runs" / "DST_Baseline_h100" / "checkpoints" / "best_model.pt"


@lru_cache(maxsize=1)
def load_dst_model(footpass_root: str, checkpoint: str, device: str = "cuda:0") -> Any:
    import torch

    module_path = Path(footpass_root) / "models" / "model_DST.py"
    spec = importlib.util.spec_from_file_location("footpass_model_dst", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.DST_Logits2Events(
        framespan=DST_FRAMESPAN,
        input_dim=DST_INPUT_DIM,
        output_dim=DST_OUTPUT_DIM,
        hidden_dim=512,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
        dropout=0.1,
    )
    payload = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def footpass_direction_flag(left_to_right: bool) -> int:
    """FOOTPASS ``left_to_right`` flag: 0 attacks the right goal (x=1)."""

    return 0 if left_to_right else 1


def slot_index(flag: int, role_id: int) -> int:
    if flag not in (0, 1) or not 1 <= role_id <= ROLES_PER_TEAM:
        raise ValueError("invalid FOOTPASS side or role")
    return flag * ROLES_PER_TEAM + role_id - 1


@dataclass(frozen=True)
class SlotAssignment:
    track_id: str
    slot: int
    first: int
    last: int

    @property
    def flag(self) -> int:
        return self.slot // ROLES_PER_TEAM

    @property
    def role_id(self) -> int:
        return self.slot % ROLES_PER_TEAM + 1


def _attack_view(x: float, y: float, flag: int) -> tuple[float, float]:
    return (x, y) if flag == 0 else (1.0 - x, 1.0 - y)


def _track_positions(
    frames: Sequence[FrameObservation],
) -> dict[str, dict[int, tuple[float, float]]]:
    positions: dict[str, dict[int, tuple[float, float]]] = {}
    for local, frame in enumerate(frames):
        for item in frame.objects:
            if (
                item.object_type in PLAYER_TYPES
                and item.track_id is not None
                and item.pitch_x is not None
                and item.pitch_y is not None
            ):
                positions.setdefault(item.track_id, {})[local] = (
                    item.pitch_x / PITCH_LENGTH,
                    item.pitch_y / PITCH_WIDTH,
                )
    return positions


def _track_teams(frames: Sequence[FrameObservation]) -> dict[str, str]:
    votes: dict[str, dict[str, int]] = {}
    for frame in frames:
        for item in frame.objects:
            if item.object_type in PLAYER_TYPES and item.track_id is not None and item.team:
                counter = votes.setdefault(item.track_id, {})
                counter[item.team] = counter.get(item.team, 0) + 1
    return {
        track: max(sorted(counter), key=lambda team: counter[team])
        for track, counter in votes.items()
    }


def assign_role_slots(
    frames: Sequence[FrameObservation],
    team_directions: Mapping[str, bool],
    *,
    goalkeeper_share: Mapping[str, float] | None = None,
    min_frames: int = 10,
    goalkeeper_min_share: float = 0.5,
) -> tuple[SlotAssignment, ...]:
    """Give each long-enough track one FOOTPASS role slot for the segment.

    Teams shift as a block, so a track is matched on its offset from its
    side's visible centroid. Tracks that do not overlap in time may share a
    slot (a broken track picked up again under a new ID).
    """

    positions = _track_positions(frames)
    teams = _track_teams(frames)
    shares = goalkeeper_share or {}
    flags = {team: footpass_direction_flag(direction) for team, direction in team_directions.items()}
    taken: dict[int, list[tuple[int, int]]] = {}
    result: list[SlotAssignment] = []

    def free(slot: int, first: int, last: int) -> bool:
        return all(last < start or first > end for start, end in taken.get(slot, ()))

    def take(track: str, slot: int, first: int, last: int) -> None:
        taken.setdefault(slot, []).append((first, last))
        result.append(SlotAssignment(track_id=track, slot=slot, first=first, last=last))

    ordered = sorted(
        (track for track, seen in positions.items() if len(seen) >= min_frames),
        key=lambda track: (-len(positions[track]), track),
    )
    keepers = [track for track in ordered if shares.get(track, 0.0) >= goalkeeper_min_share]
    for track in keepers:
        seen = positions[track]
        mean_x = sum(point[0] for point in seen.values()) / len(seen)
        flag = 0 if mean_x < 0.5 else 1
        first, last = min(seen), max(seen)
        slot = slot_index(flag, GOALKEEPER_ROLE)
        if free(slot, first, last):
            take(track, slot, first, last)

    outfield = [track for track in ordered if track not in keepers and teams.get(track) in flags]
    template_centroid = (
        sum(ROLE_TEMPLATE[role][0] for role in OUTFIELD_ROLES) / len(OUTFIELD_ROLES),
        sum(ROLE_TEMPLATE[role][1] for role in OUTFIELD_ROLES) / len(OUTFIELD_ROLES),
    )
    means: dict[str, tuple[float, float]] = {}
    for track in outfield:
        flag = flags[teams[track]]
        seen = positions[track]
        views = [_attack_view(x, y, flag) for x, y in seen.values()]
        means[track] = (
            sum(point[0] for point in views) / len(views),
            sum(point[1] for point in views) / len(views),
        )
    for team, flag in sorted(flags.items()):
        members = [track for track in outfield if teams[track] == team]
        if not members:
            continue
        centroid = (
            sum(means[track][0] for track in members) / len(members),
            sum(means[track][1] for track in members) / len(members),
        )
        shift = (centroid[0] - template_centroid[0], centroid[1] - template_centroid[1])
        for track in members:
            seen = positions[track]
            first, last = min(seen), max(seen)
            x, y = means[track][0] - shift[0], means[track][1] - shift[1]
            ranked = sorted(
                OUTFIELD_ROLES,
                key=lambda role: (ROLE_TEMPLATE[role][0] - x) ** 2 + (ROLE_TEMPLATE[role][1] - y) ** 2,
            )
            for role in ranked:
                slot = slot_index(flag, role)
                if free(slot, first, last):
                    take(track, slot, first, last)
                    break
    return tuple(sorted(result, key=lambda item: (item.slot, item.first, item.track_id)))


def build_dst_inputs(
    frames: Sequence[FrameObservation],
    assignments: Sequence[SlotAssignment],
    taad: SegmentTaadOutput,
    *,
    smoothing: int = 5,
) -> tuple[Any, Any]:
    """Return team vector (5, 26, T) and TAAD logits (9, 26, T) like FOOTPASS."""

    import numpy as np

    length = len(frames)
    positions = _track_positions(frames)
    teamvec = np.full((5, DST_SLOTS, length), MISSING_VALUE, dtype=np.float32)
    empty = np.asarray(taad.empty_logits, dtype=np.float32).reshape(-1, 1)
    logits = np.repeat(np.repeat(empty[:, None, :], DST_SLOTS, axis=1), length, axis=2)
    kernel = np.ones(smoothing) / smoothing
    for assignment in assignments:
        seen = positions.get(assignment.track_id, {})
        if not seen:
            continue
        track = np.full((length, 2), np.nan)
        for local, point in seen.items():
            track[local] = point
        observed = ~np.isnan(track[:, 0])
        smoothed = track.copy()
        for axis in range(2):
            values = np.where(observed, track[:, axis], 0.0)
            weights = np.convolve(observed.astype(float), kernel, mode="same")
            averaged = np.convolve(values, kernel, mode="same")
            with np.errstate(invalid="ignore", divide="ignore"):
                smoothed[:, axis] = np.where(observed, averaged / weights, np.nan)
        velocity = np.gradient(smoothed, axis=0) if length > 1 else np.zeros_like(smoothed)
        velocity = np.nan_to_num(velocity, nan=0.0) * FOOTPASS_VELOCITY_SCALE
        slot = assignment.slot
        frames_on = np.where(observed)[0]
        teamvec[0, slot, frames_on] = track[frames_on, 0]
        teamvec[1, slot, frames_on] = track[frames_on, 1]
        teamvec[2, slot, frames_on] = velocity[frames_on, 0]
        teamvec[3, slot, frames_on] = velocity[frames_on, 1]
        teamvec[4, slot, frames_on] = 1.0
        track_logits = taad.logits.get(assignment.track_id)
        track_present = taad.present.get(assignment.track_id)
        if track_logits is not None and track_present is not None:
            span = np.where(track_present[:length])[0]
            logits[:, slot, span] = track_logits[:, span]
    return teamvec, logits


def _one_hot_frames(frame_numbers: Any, span: int) -> Any:
    import torch

    batch, steps = frame_numbers.shape
    one_hot = torch.zeros(batch, steps, span + 2, dtype=torch.float32)
    values = frame_numbers.long()
    valid = (values >= 0) & (values <= span)
    index = torch.where(valid, values, torch.full_like(values, span + 1))
    one_hot.scatter_(2, index.unsqueeze(-1), 1.0)
    return one_hot


def run_dst(
    model: Any,
    teamvec: Any,
    logits: Any,
    *,
    device: str = "cuda:0",
) -> list[tuple[int, int, int, float]]:
    """Decode (local_frame, slot, TAAD class id, score) events, 750 frames at a time."""

    import numpy as np
    import torch

    length = teamvec.shape[2]
    rows = np.concatenate(
        [
            np.transpose(teamvec, (2, 0, 1)).reshape(length, -1),
            np.transpose(logits, (2, 0, 1)).reshape(length, -1),
        ],
        axis=1,
    ).astype(np.float32)
    events: list[tuple[int, int, int, float]] = []
    for start in range(0, length, DST_FRAMESPAN):
        chunk = rows[start : start + DST_FRAMESPAN]
        steps = len(chunk)
        padded = np.zeros((DST_FRAMESPAN, DST_INPUT_DIM), dtype=np.float32)
        padded[:steps] = chunk
        frame_numbers = np.full((1, DST_FRAMESPAN), -1.0, dtype=np.float32)
        frame_numbers[0, :steps] = np.arange(1, steps + 1)
        frames_tensor = torch.from_numpy(frame_numbers)
        source = torch.cat(
            [torch.from_numpy(padded)[None], _one_hot_frames(frames_tensor, DST_FRAMESPAN)],
            dim=-1,
        )
        with torch.no_grad():
            actions, roles, timestamps = model.forward_autoregressive(
                src=source.to(device),
                src_frames=frames_tensor.to(device),
                src_key_padding_mask=(frames_tensor == -1).to(device),
                max_tgt_len=DST_MAX_EVENTS,
            )
        action_ids = actions[0].argmax(-1).cpu().numpy()
        scores = actions[0].max(-1)[0].cpu().numpy()
        role_ids = roles[0].argmax(-1).cpu().numpy()
        frame_ids = timestamps[0].argmax(-1).cpu().numpy()
        if frame_ids.sum() == 0 or not (action_ids == 9).any():
            continue
        for step in range(DST_MAX_EVENTS):
            action = int(action_ids[step])
            if action in (8, 9):
                break
            slot = int(role_ids[step])
            local = start + int(frame_ids[step]) - 1
            if slot < DST_SLOTS and 0 <= local < length:
                events.append((local, slot, action + 1, float(scores[step])))
    return events


def _occupant(
    assignments: Sequence[SlotAssignment],
    slot: int,
    local: int,
    positions: Mapping[str, Mapping[int, tuple[float, float]]],
    tolerance: int = 12,
) -> SlotAssignment | None:
    candidates = [
        item
        for item in assignments
        if item.slot == slot and item.first - tolerance <= local <= item.last + tolerance
    ]
    if not candidates:
        return None

    def distance(item: SlotAssignment) -> int:
        seen = positions.get(item.track_id, {})
        return min((abs(frame - local) for frame in seen), default=tolerance + 1)

    return min(candidates, key=lambda item: (distance(item), item.track_id))


def dst_segment_detections(
    events: Sequence[tuple[int, int, int, float]],
    frames: Sequence[FrameObservation],
    assignments: Sequence[SlotAssignment],
    *,
    team_by_flag: Mapping[int, str],
    segment_index: int,
) -> list[TaadDetection]:
    positions = _track_positions(frames)
    detections: list[TaadDetection] = []
    for local, slot, class_id, score in events:
        occupant = _occupant(assignments, slot, local, positions)
        flag = slot // ROLES_PER_TEAM
        frame = frames[local]
        pitch_x = pitch_y = None
        if occupant is not None:
            seen = positions.get(occupant.track_id, {})
            if seen:
                nearest = min(seen, key=lambda value: abs(value - local))
                pitch_x = seen[nearest][0] * PITCH_LENGTH
                pitch_y = seen[nearest][1] * PITCH_WIDTH
        detections.append(
            TaadDetection(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                action=TAAD_CLASS_NAMES[class_id],
                score=score,
                track_id=occupant.track_id if occupant is not None else f"dst-slot-{slot}",
                team=team_by_flag.get(flag),
                pitch_x=pitch_x,
                pitch_y=pitch_y,
                segment_index=segment_index,
            )
        )
    return detections


@dataclass(frozen=True)
class DstHighlightResult:
    detections: tuple[TaadDetection, ...]
    highlights: tuple[HighlightAttack, ...]
    decoded: AttackV2DecodeResult
    assignments: Mapping[int, tuple[SlotAssignment, ...]]
    checkpoint: str


def extract_dst_highlights(
    frames: Sequence[FrameObservation],
    segments: Sequence[WideSegment],
    taad_outputs: Sequence[SegmentTaadOutput],
    taad_highlights: Sequence[HighlightAttack],
    *,
    match_id: str,
    fps: float,
    device: str = "cuda:0",
    footpass_root: Path | None = None,
    checkpoint: Path | None = None,
) -> DstHighlightResult:
    """Re-read each TAAD segment with DST; team sides come from the TAAD finish."""

    root = footpass_root or resolve_footpass_root()
    checkpoint_path = checkpoint or default_dst_checkpoint(root)
    model = load_dst_model(str(root), str(checkpoint_path), device)
    detections: list[TaadDetection] = []
    assignments_by_segment: dict[int, tuple[SlotAssignment, ...]] = {}
    for segment_index, (segment, output) in enumerate(zip(segments, taad_outputs)):
        highlight = next(
            (item for item in taad_highlights if item.segment_index == segment_index), None
        )
        directions = highlight.team_directions() if highlight is not None else {}
        if not directions:
            continue
        window = frames[segment.start_index : segment.end_index + 1]
        assignments = assign_role_slots(
            window,
            directions,
            goalkeeper_share=_goalkeeper_share(frames, segment),
        )
        assignments_by_segment[segment_index] = assignments
        teamvec, logits = build_dst_inputs(window, assignments, output)
        events = run_dst(model, teamvec, logits, device=device)
        team_by_flag = {footpass_direction_flag(value): team for team, value in directions.items()}
        detections.extend(
            dst_segment_detections(
                events,
                window,
                assignments,
                team_by_flag=team_by_flag,
                segment_index=segment_index,
            )
        )
    if assignments_by_segment and not detections:
        raise RuntimeError(
            "DST không giải mã được sự kiện nào trên clip này. DST chỉ được train trên cửa sổ "
            "750 frame có đủ 22 cầu thủ; trên FOOTPASS val, chỉ cần bỏ bớt cầu thủ hoặc đệm "
            "cửa sổ là decoder lặp một token ở frame 0 và không ra EOS. Video phát sóng chỉ "
            "thấy khoảng 10 cầu thủ, nên cần train lại DST với cầu thủ bị che ngẫu nhiên."
        )
    detections.sort(key=lambda item: (item.frame_index, -item.score))
    highlights = highlight_attack_events(
        detections, segments, frames, match_id=match_id, source="footpass_dst"
    )
    decoded = decode_highlight_attacks(highlights, match_id=match_id, fps=fps)
    return DstHighlightResult(
        detections=tuple(detections),
        highlights=highlights,
        decoded=decoded,
        assignments=assignments_by_segment,
        checkpoint=str(checkpoint_path),
    )


__all__ = [
    "DST_FRAMESPAN",
    "DstHighlightResult",
    "FOOTPASS_VELOCITY_SCALE",
    "ROLE_TEMPLATE",
    "SlotAssignment",
    "assign_role_slots",
    "build_dst_inputs",
    "default_dst_checkpoint",
    "dst_segment_detections",
    "extract_dst_highlights",
    "footpass_direction_flag",
    "load_dst_model",
    "run_dst",
    "slot_index",
]
