from __future__ import annotations

import binascii
import bz2
import io
import json
import lzma
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


GAMESTATE_TASK = "gamestate-2024"
GAMESTATE_ARCHIVE_URLS = {
    "train": "https://exrcsdrive.kaust.edu.sa/public.php/webdav/train.zip",
    "valid": "https://exrcsdrive.kaust.edu.sa/public.php/webdav/valid.zip",
    "test": "https://exrcsdrive.kaust.edu.sa/public.php/webdav/test.zip",
    "challenge": "https://exrcsdrive.kaust.edu.sa/public.php/webdav/challenge.zip",
}
# These are the public share credentials published by SoccerNet's Downloader.py.
GAMESTATE_PUBLIC_USER = "iOJmJH6rYnx7mOS"
GAMESTATE_PUBLIC_PASSWORD = "SoccerNet"
PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0


class SoccerNetDownloadError(RuntimeError):
    pass


class CurlRangeReader(io.RawIOBase):
    """Seekable HTTP reader backed by verified curl range requests."""

    def __init__(self, url: str, username: str, password: str) -> None:
        super().__init__()
        executable = shutil.which("curl") or shutil.which("curl.exe")
        if executable is None:
            raise SoccerNetDownloadError("curl is required for partial SoccerNet downloads")
        self._curl = executable
        self._url = url
        self._auth = f"{username}:{password}"
        self._position = 0
        self._size = self._read_size()

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = min(position, self._size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._position
        if size == 0:
            return b""
        end = min(self._position + size, self._size) - 1
        data = self.fetch_range(self._position, end)
        self._position += len(data)
        return data

    def fetch_range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self._size:
            raise ValueError(f"invalid byte range {start}-{end} for {self._size}-byte file")
        chunk_bytes = 2 * 1024 * 1024
        spans: list[tuple[int, int]] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + chunk_bytes - 1, end)
            spans.append((cursor, chunk_end))
            cursor = chunk_end + 1
        if len(spans) == 1:
            return self._fetch_range_once(*spans[0])
        parts: list[bytes | None] = [None] * len(spans)

        def load(item: tuple[int, tuple[int, int]]) -> None:
            index, span = item
            parts[index] = self._fetch_range_once(*span)
            print(
                f"SoccerNet range {index + 1}/{len(spans)} "
                f"({span[1] - span[0] + 1} bytes)",
                file=sys.stderr,
                flush=True,
            )

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(load, enumerate(spans)))
        return b"".join(part for part in parts if part is not None)

    def _fetch_range_once(self, start: int, end: int) -> bytes:
        command = self._base_command() + ["--range", f"{start}-{end}", self._url]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise SoccerNetDownloadError(f"SoccerNet range request failed: {message}")
        expected = end - start + 1
        if len(result.stdout) != expected:
            raise SoccerNetDownloadError(
                f"range request returned {len(result.stdout)} bytes; expected {expected}"
            )
        return result.stdout

    def _base_command(self) -> list[str]:
        return [
            self._curl,
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--connect-timeout",
            "20",
            "--retry",
            "3",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--speed-limit",
            "2048",
            "--speed-time",
            "30",
            "--max-time",
            "120",
            "--user",
            self._auth,
        ]

    def _read_size(self) -> int:
        command = self._base_command() + ["--head", self._url]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise SoccerNetDownloadError(f"cannot inspect SoccerNet archive: {message}")
        headers = result.stdout.decode("iso-8859-1", errors="replace")
        matches = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", headers)
        if not matches:
            raise SoccerNetDownloadError("server did not return a Content-Length header")
        return int(matches[-1])


def _safe_destination(root: Path, archive_name: str) -> Path:
    relative = PurePosixPath(archive_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise SoccerNetDownloadError(f"unsafe ZIP member path: {archive_name}")
    destination = root.joinpath(*relative.parts).resolve()
    root_resolved = root.resolve()
    if root_resolved != destination and root_resolved not in destination.parents:
        raise SoccerNetDownloadError(f"ZIP member escapes destination: {archive_name}")
    return destination


def _decompress_member(info: zipfile.ZipInfo, compressed: bytes) -> bytes:
    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif info.compress_type == zipfile.ZIP_BZIP2:
        data = bz2.decompress(compressed)
    elif info.compress_type == zipfile.ZIP_LZMA:
        data = lzma.decompress(compressed)
    else:
        raise SoccerNetDownloadError(
            f"unsupported ZIP compression {info.compress_type} for {info.filename}"
        )
    if len(data) != info.file_size:
        raise SoccerNetDownloadError(f"size check failed for {info.filename}")
    if (binascii.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise SoccerNetDownloadError(f"CRC check failed for {info.filename}")
    return data


def _extract_contiguous_members(
    reader: CurlRangeReader,
    all_infos: list[zipfile.ZipInfo],
    selected: list[zipfile.ZipInfo],
    destination: Path,
    central_directory_offset: int,
) -> int:
    ordered = sorted(all_infos, key=lambda item: item.header_offset)
    selected_names = {item.filename for item in selected}
    indexes = [index for index, item in enumerate(ordered) if item.filename in selected_names]
    if not indexes or indexes != list(range(indexes[0], indexes[-1] + 1)):
        raise SoccerNetDownloadError("sequence files are not contiguous in the remote archive")

    start = ordered[indexes[0]].header_offset
    after_index = indexes[-1] + 1
    end = (
        ordered[after_index].header_offset
        if after_index < len(ordered)
        else central_directory_offset
    )
    blob = reader.fetch_range(start, end - 1)

    for info in sorted(selected, key=lambda item: item.header_offset):
        target = _safe_destination(destination, info.filename)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        offset = info.header_offset - start
        header = blob[offset : offset + 30]
        if len(header) != 30:
            raise SoccerNetDownloadError(f"truncated local header for {info.filename}")
        signature, *fields = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise SoccerNetDownloadError(f"invalid local header for {info.filename}")
        filename_length, extra_length = fields[-2:]
        data_start = offset + 30 + filename_length + extra_length
        data_end = data_start + info.compress_size
        data = _decompress_member(info, blob[data_start:data_end])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return len(blob)


def download_gamestate_sequence(
    output_directory: Path,
    split: str = "valid",
    sequence: str = "first",
) -> dict[str, object]:
    if split not in GAMESTATE_ARCHIVE_URLS:
        raise ValueError(f"unsupported split: {split}")
    output_directory = output_directory.resolve()
    split_directory = output_directory / GAMESTATE_TASK / split
    reader = CurlRangeReader(
        GAMESTATE_ARCHIVE_URLS[split], GAMESTATE_PUBLIC_USER, GAMESTATE_PUBLIC_PASSWORD
    )
    try:
        with zipfile.ZipFile(reader) as archive:
            infos = archive.infolist()
            central_directory_offset = archive.start_dir
        roots: list[str] = []
        for info in infos:
            root = PurePosixPath(info.filename).parts[0]
            if root.startswith("SNGS-") and root not in roots:
                roots.append(root)
        if not roots:
            raise SoccerNetDownloadError("no SNGS sequence found in archive")
        selected_sequence = roots[0] if sequence == "first" else sequence.upper()
        if selected_sequence not in roots:
            raise SoccerNetDownloadError(
                f"sequence {selected_sequence!r} is not present in split {split}"
            )
        selected = [
            info
            for info in infos
            if info.filename == selected_sequence
            or info.filename.startswith(f"{selected_sequence}/")
        ]
        sequence_directory = split_directory / selected_sequence
        label_path = sequence_directory / "Labels-GameState.json"
        images = sorted((sequence_directory / "img1").glob("*.jpg"))
        if label_path.exists() and images:
            downloaded_bytes = 0
            cached = True
        else:
            split_directory.mkdir(parents=True, exist_ok=True)
            downloaded_bytes = _extract_contiguous_members(
                reader, infos, selected, split_directory, central_directory_offset
            )
            images = sorted((sequence_directory / "img1").glob("*.jpg"))
            cached = False
    finally:
        reader.close()

    manifest: dict[str, object] = {
        "source": "SoccerNet Game State Reconstruction",
        "task": GAMESTATE_TASK,
        "split": split,
        "sequence": selected_sequence,
        "sequence_selection": "first archive sequence" if sequence == "first" else "explicit",
        "archive_url": GAMESTATE_ARCHIVE_URLS[split],
        "archive_size_bytes": reader.size,
        "downloaded_range_bytes": downloaded_bytes,
        "cached": cached,
        "sequence_directory": str(sequence_directory),
        "labels": str(label_path),
        "image_count": len(images),
        "first_image": str(images[0]) if images else None,
        "last_image": str(images[-1]) if images else None,
    }
    manifest_path = sequence_directory / "download-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def _frame_number(image: dict[str, Any]) -> int | None:
    for key in ("frame_id", "frame", "frame_index"):
        value = image.get(key)
        if isinstance(value, int):
            return value
    file_name = image.get("file_name")
    if isinstance(file_name, str):
        match = re.search(r"(\d+)(?=\.[^.]+$)", file_name)
        if match:
            return int(match.group(1))
    return None


def _pitch_point(annotation: dict[str, Any]) -> tuple[float, float] | None:
    pitch = annotation.get("bbox_pitch")
    if not isinstance(pitch, dict):
        return None
    x = pitch.get("x_bottom_middle")
    y = pitch.get("y_bottom_middle")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if math.isfinite(float(x)) and math.isfinite(float(y)):
            return float(x), float(y)
    return None


def _inside_pitch_margin(x: float, y: float, margin_m: float) -> bool:
    return (
        abs(x) <= PITCH_HALF_LENGTH_M + margin_m
        and abs(y) <= PITCH_HALF_WIDTH_M + margin_m
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _provisional_attack_assessment(
    info: dict[str, Any],
    ball_by_frame: dict[int, tuple[float, float]],
    players_by_frame: dict[int, list[tuple[str, float, float]]],
    pitch_margin_m: float,
) -> dict[str, object]:
    valid_ball = [
        (frame, point)
        for frame, point in sorted(ball_by_frame.items())
        if _inside_pitch_margin(point[0], point[1], pitch_margin_m)
    ]
    if not valid_ball:
        return {
            "status": "insufficient_context",
            "effective": None,
            "reason": "No valid ball trajectory is available for attack assessment.",
        }

    early_limit = valid_ball[0][0] + max(1, round((valid_ball[-1][0] - valid_ball[0][0]) * 0.4))
    early_team_votes: Counter[str] = Counter()
    possession_votes: Counter[str] = Counter()
    for frame, (ball_x, ball_y) in valid_ball:
        candidates = players_by_frame.get(frame, [])
        if not candidates:
            continue
        team, player_x, player_y = min(
            candidates, key=lambda item: math.hypot(item[1] - ball_x, item[2] - ball_y)
        )
        distance = math.hypot(player_x - ball_x, player_y - ball_y)
        if distance <= 3.0:
            possession_votes[team] += 1
        if frame <= early_limit and distance <= 10.0:
            early_team_votes[team] += 1

    if not early_team_votes:
        return {
            "status": "insufficient_context",
            "effective": None,
            "reason": "The attacking team cannot be inferred near the start of the clip.",
        }
    attacking_team = early_team_votes.most_common(1)[0][0]
    attack_direction = "left" if attacking_team == "right" else "right"
    start_x = _median([point[0] for _, point in valid_ball[: min(25, len(valid_ball))]])
    extreme_x = (
        min(point[0] for _, point in valid_ball)
        if attack_direction == "left"
        else max(point[0] for _, point in valid_ball)
    )
    progression = start_x - extreme_x if attack_direction == "left" else extreme_x - start_x
    reached_final_third = extreme_x <= -17.5 if attack_direction == "left" else extreme_x >= 17.5
    reached_box = extreme_x <= -36.0 if attack_direction == "left" else extreme_x >= 36.0
    action_class = str(info.get("action_class") or "unknown")
    shot_observed = action_class.lower() in {"shot", "shot on target", "goal"}
    score = min(max(progression, 0.0) / 40.0, 1.0) * 30.0
    score += 20.0 if reached_final_third else 0.0
    score += 20.0 if reached_box else 0.0
    score += 30.0 if shot_observed else 0.0
    effective = score >= 50.0
    return {
        "status": "provisional_effective" if effective else "provisional_ineffective_chance_creation",
        "effective": effective,
        "confidence": "medium",
        "heuristic_score": round(score, 1),
        "attacking_team": attacking_team,
        "attack_direction": attack_direction,
        "start_ball_x_m": round(start_x, 3),
        "furthest_ball_x_m": round(extreme_x, 3),
        "max_progression_m": round(max(progression, 0.0), 3),
        "reached_final_third": reached_final_third,
        "reached_penalty_area": reached_box,
        "shot_observed": shot_observed,
        "estimated_sequence_xg": 0.0 if not shot_observed else None,
        "terminal_action": action_class,
        "possession_frames_by_team": dict(sorted(possession_votes.items())),
        "reason": (
            "The move progressed substantially but produced no final-third entry, "
            "penalty-area entry, or shot before the annotated terminal action."
            if not effective
            else "The move reached high-value attacking zones or produced a shot."
        ),
        "limitations": [
            "team is inferred from nearest players at the start",
            "the Foul label does not identify which team received the free kick",
            "this is a deterministic research baseline, not a trained xG model",
        ],
    }


def analyze_gamestate_sequence(
    sequence_directory: Path,
    fps: float | None = None,
    max_speed_mps: float = 45.0,
    pitch_margin_m: float = 5.0,
) -> dict[str, object]:
    if max_speed_mps <= 0:
        raise ValueError("max_speed_mps must be positive")
    if pitch_margin_m < 0:
        raise ValueError("pitch_margin_m must be non-negative")
    sequence_directory = sequence_directory.resolve()
    label_path = sequence_directory / "Labels-GameState.json"
    if not label_path.exists():
        raise FileNotFoundError(f"missing SoccerNet labels: {label_path}")
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Labels-GameState.json must contain images and annotations lists")

    info = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
    effective_fps = float(fps if fps is not None else info.get("frame_rate", 25.0))
    image_frames: dict[object, int | None] = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = image.get("id", image.get("image_id"))
        if image_id is not None:
            image_frames[image_id] = _frame_number(image)
    categories = {
        category.get("id"): category
        for category in payload.get("categories", [])
        if isinstance(category, dict) and category.get("id") is not None
    }
    roles: Counter[str] = Counter()
    teams: Counter[str] = Counter()
    tracks: set[object] = set()
    jersey_numbers: set[object] = set()
    pitch_count = 0
    object_annotations = 0
    pitch_points_by_role: Counter[str] = Counter()
    in_field_points_by_role: Counter[str] = Counter()
    trajectories: dict[object, list[tuple[int, float, float, str]]] = defaultdict(list)
    ball_by_frame: dict[int, tuple[float, float]] = {}
    players_by_frame: dict[int, list[tuple[str, float, float]]] = defaultdict(list)

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        attributes = annotation.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        category = categories.get(annotation.get("category_id"), {})
        is_object = category.get("supercategory") == "object" or (
            not category and annotation.get("track_id") is not None
        )
        if not is_object:
            continue
        object_annotations += 1
        role = str(
            annotation.get("role")
            or attributes.get("role")
            or category.get("name")
            or "unknown"
        )
        team = str(annotation.get("team") or attributes.get("team") or "unknown")
        roles[role] += 1
        teams[team] += 1
        track_id = annotation.get("track_id")
        if track_id is not None:
            tracks.add(track_id)
        jersey = annotation.get("jersey_number", attributes.get("jersey"))
        if jersey is not None:
            jersey_numbers.add(jersey)
        point = _pitch_point(annotation)
        frame = image_frames.get(annotation.get("image_id"))
        if point is not None:
            pitch_count += 1
            pitch_points_by_role[role] += 1
            if _inside_pitch_margin(point[0], point[1], pitch_margin_m):
                in_field_points_by_role[role] += 1
            if track_id is not None and frame is not None:
                trajectories[track_id].append((frame, point[0], point[1], role))
            if frame is not None and role == "ball":
                ball_by_frame[frame] = point
            elif frame is not None and role in {"player", "goalkeeper"} and team != "unknown":
                players_by_frame[frame].append((team, point[0], point[1]))

    speeds_by_role: dict[str, list[float]] = defaultdict(list)
    raw_speeds_by_role: dict[str, list[float]] = defaultdict(list)
    in_field_speed_samples: Counter[str] = Counter()
    rejected_speed_samples: Counter[str] = Counter()
    for samples in trajectories.values():
        samples.sort()
        for previous, current in zip(samples, samples[1:]):
            frame_gap = current[0] - previous[0]
            if frame_gap <= 0 or frame_gap > 5:
                continue
            distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
            role = current[3]
            speed = distance * effective_fps / frame_gap
            raw_speeds_by_role[role].append(speed)
            points_in_field = _inside_pitch_margin(
                previous[1], previous[2], pitch_margin_m
            ) and _inside_pitch_margin(current[1], current[2], pitch_margin_m)
            if not points_in_field:
                continue
            in_field_speed_samples[role] += 1
            if speed > max_speed_mps:
                rejected_speed_samples[role] += 1
                continue
            speeds_by_role[role].append(speed)

    speed_summary = {
        role: {
            "samples": len(values),
            "mean_mps": round(sum(values) / len(values), 3),
            "p95_mps": round(_percentile(values, 0.95), 3),
            "max_mps": round(max(values), 3),
            "raw_samples": len(raw_speeds_by_role[role]),
            "in_field_samples": in_field_speed_samples[role],
            "rejected_over_limit": rejected_speed_samples[role],
            "raw_max_mps": round(max(raw_speeds_by_role[role]), 3),
        }
        for role, values in sorted(speeds_by_role.items())
        if values
    }
    coordinate_quality = {
        role: {
            "points": count,
            "inside_pitch_margin": in_field_points_by_role[role],
            "inside_ratio": round(in_field_points_by_role[role] / count, 4),
        }
        for role, count in sorted(pitch_points_by_role.items())
    }
    frame_numbers = [frame for frame in image_frames.values() if frame is not None]
    report: dict[str, object] = {
        "source": str(label_path),
        "sequence": sequence_directory.name,
        "dataset_version": info.get("version"),
        "fps": effective_fps,
        "fps_source": "cli_override" if fps is not None else "dataset_info",
        "frames": len(images),
        "duration_sec": round(len(images) / effective_fps, 3),
        "frame_range": [min(frame_numbers), max(frame_numbers)] if frame_numbers else None,
        "annotations": len(annotations),
        "object_annotations": object_annotations,
        "unique_tracks": len(tracks),
        "roles": dict(sorted(roles.items())),
        "teams": dict(sorted(teams.items())),
        "known_jersey_numbers": sorted(jersey_numbers, key=str),
        "pitch_coordinate_coverage": round(pitch_count / object_annotations, 4)
        if object_annotations
        else 0.0,
        "speed_by_role": speed_summary,
        "motion_quality": {
            "pitch_dimensions_m": [105.0, 68.0],
            "pitch_margin_m": pitch_margin_m,
            "max_accepted_speed_mps": max_speed_mps,
            "coordinates_by_role": coordinate_quality,
        },
        "clip_context": {
            "action_class": info.get("action_class"),
            "action_position_ms": info.get("action_position"),
            "game_time_start": info.get("game_time_start"),
            "game_time_stop": info.get("game_time_stop"),
            "visibility": info.get("visibility"),
        },
        "attack_assessment": _provisional_attack_assessment(
            info, ball_by_frame, players_by_frame, pitch_margin_m
        ),
    }
    report_path = sequence_directory / "analysis-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report
