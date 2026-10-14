"""Video estimate of whether a shot was scored: scoreboard OCR plus ball-in-goal.

The result is a video estimate, never a gold label. Scoreboard sides cannot be
mapped to the tracked team ids, so a score increase is attributed to the
attacking team of the shot that precedes it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .attack_v2_shot_insights import actions_with_resolved_outcome
from .models import AttackSequenceV2

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_HALF_WIDTH_M = 3.66
SCOREBOARD_CACHE_VERSION = 4

_DIGIT_LOOKALIKES = str.maketrans({"T": "1", "I": "1", "l": "1", "|": "1", "i": "1", "O": "0", "o": "0"})
_TEAM_NAME = re.compile(r"[A-Z][A-Z0-9]{1,3}")
_SCORE = re.compile(r"\d{1,2}")
_JOINED_SCORE = re.compile(r"^(\w{1,2})\s*[-:–]\s*(\w{1,2})$")


@dataclass(frozen=True)
class ScoreReading:
    timestamp_sec: float
    home: int
    away: int
    text: str = ""

    @property
    def total(self) -> int:
        return self.home + self.away

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_sec": self.timestamp_sec,
            "home": self.home,
            "away": self.away,
            "text": self.text,
        }


@dataclass(frozen=True)
class ShotOutcomeEvidence:
    shot_sec: float
    outcome: str
    source: str | None
    score_before: tuple[int, int] | None = None
    score_after: tuple[int, int] | None = None
    score_change_sec: float | None = None
    ball_in_goal_sec: tuple[float, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_goal(self) -> bool:
        return self.outcome == "goal"

    def summary(self) -> str:
        parts: list[str] = []
        if self.score_before is not None:
            before = f"{self.score_before[0]}-{self.score_before[1]}"
            if self.score_after is not None and self.score_after != self.score_before:
                change = (
                    f" lúc {self.score_change_sec:.0f}s" if self.score_change_sec is not None else ""
                )
                parts.append(
                    f"tỷ số {before} → {self.score_after[0]}-{self.score_after[1]}{change}"
                )
            elif self.outcome == "no_goal":
                parts.append(f"tỷ số giữ {before}")
            else:
                parts.append(f"tỷ số trước cú sút {before}")
        if self.ball_in_goal_sec:
            parts.append(f"bóng trong khung thành {len(self.ball_in_goal_sec)} frame")
        parts.extend(self.notes)
        return "; ".join(parts) if parts else "không có bằng chứng"

    def to_dict(self) -> dict[str, object]:
        return {
            "shot_sec": self.shot_sec,
            "outcome": self.outcome,
            "source": self.source,
            "score_before": None if self.score_before is None else list(self.score_before),
            "score_after": None if self.score_after is None else list(self.score_after),
            "score_change_sec": self.score_change_sec,
            "ball_in_goal_sec": list(self.ball_in_goal_sec),
            "notes": list(self.notes),
            "summary": self.summary(),
        }


def _token_left(token: Any) -> float:
    box = token[0]
    try:
        return float(min(point[0] for point in box))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _as_score(text: str) -> int | None:
    value = text.strip()
    if len(value) <= 2:
        value = value.translate(_DIGIT_LOOKALIKES)
    return int(value) if _SCORE.fullmatch(value) else None


def _as_joined_score(text: str) -> tuple[int, int] | None:
    joined = _JOINED_SCORE.match(text.strip())
    if not joined:
        return None
    left, right = _as_score(joined.group(1)), _as_score(joined.group(2))
    return None if left is None or right is None else (left, right)


def _as_team_name(text: str) -> str | None:
    value = text.strip().upper()
    if not value.isalnum() or sum(character.isalpha() for character in value) < 1:
        return None
    return value if _TEAM_NAME.fullmatch(value) else None


@dataclass(frozen=True)
class ScoreboardAnchor:
    """Where the score bug sits: team-name boxes and one or two score boxes."""

    home_name: str
    away_name: str
    home_box: tuple[int, int, int, int]
    away_box: tuple[int, int, int, int]
    score_boxes: tuple[tuple[int, int, int, int], ...]


def _box(token: Any) -> tuple[int, int, int, int]:
    xs = [float(point[0]) for point in token[0]]
    ys = [float(point[1]) for point in token[0]]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def match_scoreboard_tokens(
    tokens: Sequence[Any],
) -> tuple[tuple[int, int], ScoreboardAnchor] | None:
    """Find ``NAME s s NAME`` (or ``NAME s-s NAME``) in OCR tokens ``(box, text, conf)``.

    Single-character lookalikes (``T``, ``I``, ``O``) are mapped to digits because
    broadcast score fonts are often misread that way.
    """

    items: list[tuple[str, Any, Any]] = []
    for token in sorted(tokens, key=_token_left):
        text = str(token[1]).strip()
        if not text or text in {"-", ":", "–"}:
            continue
        joined = _as_joined_score(text)
        if joined is not None:
            items.append(("joined", joined, token))
            continue
        score = _as_score(text)
        name = _as_team_name(text)
        if score is not None:
            items.append(("score", score, token))
        elif name is not None:
            items.append(("name", name, token))
        else:
            items.append(("other", text, token))
    for index in range(len(items)):
        kinds = [kind for kind, _value, _token in items[index : index + 4]]
        if kinds == ["name", "score", "score", "name"]:
            home, first, second, away = items[index : index + 4]
            score_boxes = (_box(first[2]), _box(second[2]))
            score = (int(first[1]), int(second[1]))
        elif kinds[:3] == ["name", "joined", "name"]:
            home, middle, away = items[index : index + 3]
            score_boxes = (_box(middle[2]),)
            score = middle[1]
        else:
            continue
        anchor = ScoreboardAnchor(
            home_name=str(home[1]),
            away_name=str(away[1]),
            home_box=_box(home[2]),
            away_box=_box(away[2]),
            score_boxes=score_boxes,
        )
        return score, anchor
    return _match_stacked_scoreboard(items)


def _match_stacked_scoreboard(
    items: Sequence[tuple[str, Any, Any]],
) -> tuple[tuple[int, int], ScoreboardAnchor] | None:
    """Bundesliga-style bug: each team name sits directly above its own score."""

    names = [(value, _box(token)) for kind, value, token in items if kind == "name"]
    scores = [(value, _box(token)) for kind, value, token in items if kind == "score"]
    columns: list[tuple[str, tuple[int, int, int, int], int, tuple[int, int, int, int]]] = []
    for name, name_box in names:
        name_height = max(1, name_box[3] - name_box[1])
        below = [
            (value, box)
            for value, box in scores
            if name_box[0] - name_height <= (box[0] + box[2]) / 2 <= name_box[2] + name_height
            and box[1] >= (name_box[1] + name_box[3]) / 2
            and box[1] - name_box[3] <= 2 * name_height
        ]
        if below:
            value, box = min(below, key=lambda item: item[1][1])
            columns.append((name, name_box, int(value), box))
    if len(columns) < 2:
        return None
    columns.sort(key=lambda item: item[1][0])
    home, away = columns[0], columns[1]
    if home[3] == away[3]:
        return None
    anchor = ScoreboardAnchor(
        home_name=home[0],
        away_name=away[0],
        home_box=home[1],
        away_box=away[1],
        score_boxes=(home[3], away[3]),
    )
    return (home[2], away[2]), anchor


def parse_scoreboard_tokens(tokens: Sequence[Any]) -> tuple[int, int] | None:
    matched = match_scoreboard_tokens(tokens)
    return None if matched is None else matched[0]


class ScoreboardReader:
    """Reads the broadcast score bug every ``step_sec`` seconds with RapidOCR (CPU)."""

    def __init__(
        self,
        *,
        step_sec: float = 1.0,
        band_height: float = 0.2,
        band_width: float = 0.5,
        rescan_every: int = 5,
        max_detections: int = 30,
        ocr: Any | None = None,
    ) -> None:
        self.step_sec = float(step_sec)
        self.band_height = float(band_height)
        self.band_width = float(band_width)
        self.rescan_every = int(rescan_every)
        self.max_detections = int(max_detections)
        self._ocr = ocr

    def _engine(self) -> Any:
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        return self._ocr

    def detect(self, frame: Any) -> tuple[tuple[int, int], ScoreboardAnchor] | None:
        """Full text detection on the top band; slow (about 1-3 s on CPU)."""

        height, width = frame.shape[:2]
        band = frame[: int(height * self.band_height), : int(width * self.band_width)]
        result = self._engine()(band)
        tokens = result[0] if isinstance(result, tuple) else result
        return match_scoreboard_tokens(list(tokens or []))

    def _recognise(self, frame: Any, box: tuple[int, int, int, int]) -> str:
        x0, y0, x1, y1 = box
        crop = frame[max(0, y0 - 3) : y1 + 3, max(0, x0 - 3) : x1 + 3]
        if crop.size == 0:
            return ""
        result = self._engine()(crop, use_det=False, use_cls=False, use_rec=True)
        lines = result[0] if isinstance(result, tuple) else result
        if not lines:
            return ""
        return str(lines[0][0]).strip()

    def read_anchored(self, frame: Any, anchor: ScoreboardAnchor) -> tuple[int, int] | None:
        """Recognition only on the anchored boxes; about 0.07 s per frame on CPU.

        A read counts only when both team names match the anchor, so replays and
        close-ups without the score bug are skipped instead of misread.
        """

        if _as_team_name(self._recognise(frame, anchor.home_box)) != anchor.home_name:
            return None
        if _as_team_name(self._recognise(frame, anchor.away_box)) != anchor.away_name:
            return None
        texts = [self._recognise(frame, box) for box in anchor.score_boxes]
        if len(texts) == 1:
            return _as_joined_score(texts[0])
        home, away = _as_score(texts[0]), _as_score(texts[1])
        return None if home is None or away is None else (home, away)

    def read_video(
        self,
        video_path: Path,
        *,
        cache_path: Path | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[ScoreReading, ...]:
        signature = {
            "version": SCOREBOARD_CACHE_VERSION,
            "step_sec": self.step_sec,
            "band": [self.band_height, self.band_width],
        }
        if cache_path is not None and cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("signature") == signature:
                return tuple(ScoreReading(**item) for item in payload.get("readings") or ())
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        times = [index * self.step_sec for index in range(int(duration / self.step_sec) + 1)]
        readings: list[ScoreReading] = []
        anchor: ScoreboardAnchor | None = None
        misses = 0
        detections = 0
        position = 0
        try:
            for done, timestamp in enumerate(times, start=1):
                target = int(round(timestamp * fps))
                ok = True
                while ok and position < target:
                    ok = capture.grab()
                    position += 1
                ok, frame = capture.read() if ok else (False, None)
                position += 1
                if anchor is None and detections >= self.max_detections:
                    break
                if ok:
                    score = None if anchor is None else self.read_anchored(frame, anchor)
                    # The bug is hidden during replays; a full rescan now and then
                    # re-anchors it if the broadcast moved or restyled it.
                    if score is None and misses % self.rescan_every == 0:
                        detected = self.detect(frame)
                        detections += 1
                        if detected is not None:
                            score, anchor = detected
                            detections = 0
                    if score is None:
                        misses += 1
                    else:
                        misses = 0
                        readings.append(
                            ScoreReading(
                                round(timestamp, 3),
                                score[0],
                                score[1],
                                f"{anchor.home_name} {score[0]}-{score[1]} {anchor.away_name}",
                            )
                        )
                if progress is not None:
                    progress(done, len(times))
        finally:
            capture.release()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {"signature": signature, "readings": [item.to_dict() for item in readings]},
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
        return tuple(readings)


def ball_in_goal_times(
    frames: Iterable[Any],
    *,
    shot_sec: float,
    left_to_right: bool | None,
    window_sec: float = 3.0,
    min_confidence: float = 0.25,
    line_margin_m: float = 0.5,
    post_margin_m: float = 1.0,
) -> tuple[float, ...]:
    """Timestamps after the shot where a ball sits behind the attacked goal line."""

    if left_to_right is None:
        return ()
    low = PITCH_WIDTH_M / 2 - GOAL_HALF_WIDTH_M - post_margin_m
    high = PITCH_WIDTH_M / 2 + GOAL_HALF_WIDTH_M + post_margin_m
    times: list[float] = []
    for frame in frames:
        timestamp = float(frame.timestamp_sec)
        if timestamp < shot_sec or timestamp > shot_sec + window_sec or frame.scene_cut:
            continue
        for item in frame.objects:
            if item.object_type != "ball" or item.pitch_x is None or item.pitch_y is None:
                continue
            if float(item.confidence) < min_confidence:
                continue
            x, y = float(item.pitch_x), float(item.pitch_y)
            if not left_to_right:
                x, y = PITCH_LENGTH_M - x, PITCH_WIDTH_M - y
            if x >= PITCH_LENGTH_M - line_margin_m and low <= y <= high:
                times.append(round(timestamp, 3))
                break
    return tuple(times)


def _modal_score(readings: Sequence[ScoreReading]) -> tuple[int, int] | None:
    if not readings:
        return None
    counts = Counter((item.home, item.away) for item in readings)
    return counts.most_common(1)[0][0]


def infer_shot_outcome(
    shot_sec: float,
    readings: Sequence[ScoreReading],
    *,
    after_limit_sec: float | None = None,
    ball_times: Sequence[float] = (),
    before_window_sec: float = 30.0,
    min_agreeing: int = 2,
    settle_sec: float = 10.0,
    min_ball_frames: int = 2,
) -> ShotOutcomeEvidence:
    """Goal if the score total rises after the shot (confirmed by ``min_agreeing`` reads)
    or, without a usable scoreboard, if the ball is seen behind the line in
    ``min_ball_frames`` frames. ``no_goal`` needs an unchanged score at least
    ``settle_sec`` after the shot; otherwise the outcome stays ``unknown``.
    ``after_limit_sec`` (usually the next shot) bounds which reads belong to this shot.
    """

    ordered = sorted(readings, key=lambda item: item.timestamp_sec)
    before_reads = [
        item
        for item in ordered
        if shot_sec - before_window_sec <= item.timestamp_sec <= shot_sec
    ]
    if not before_reads:
        earlier = [item for item in ordered if item.timestamp_sec <= shot_sec]
        before_reads = earlier[-1:]
    before = _modal_score(before_reads[-5:])
    after_reads = [
        item
        for item in ordered
        if item.timestamp_sec > shot_sec
        and (after_limit_sec is None or item.timestamp_sec < after_limit_sec)
    ]
    ball = tuple(ball_times)
    ball_goal = len(ball) >= min_ball_frames
    notes: list[str] = []

    score_outcome: str | None = None
    score_after: tuple[int, int] | None = None
    change_sec: float | None = None
    if before is not None and after_reads:
        before_total = before[0] + before[1]
        raised = Counter(
            (item.home, item.away) for item in after_reads if item.total == before_total + 1
        )
        if raised:
            candidate, count = raised.most_common(1)[0]
            if count >= min_agreeing and candidate[0] >= before[0] and candidate[1] >= before[1]:
                score_outcome = "goal"
                score_after = candidate
                change_sec = next(
                    item.timestamp_sec
                    for item in after_reads
                    if (item.home, item.away) == candidate
                )
        if score_outcome is None:
            settled = [
                item
                for item in after_reads
                if item.timestamp_sec >= shot_sec + settle_sec
                and (item.home, item.away) == before
            ]
            if settled:
                score_outcome = "no_goal"
                score_after = before
            else:
                score_after = _modal_score(after_reads)
                notes.append(
                    f"đọc được tỷ số tới {after_reads[-1].timestamp_sec:.0f}s, chưa đủ "
                    f"{settle_sec:.0f} s sau cú sút để scoreboard kịp cập nhật"
                )
    elif before is not None:
        notes.append("không đọc được scoreboard sau cú sút")
    else:
        notes.append("không đọc được scoreboard trước cú sút")

    if score_outcome == "goal":
        source = "video_scoreboard+ball" if ball_goal else "video_scoreboard"
        return ShotOutcomeEvidence(shot_sec, "goal", source, before, score_after, change_sec, ball, tuple(notes))
    if score_outcome == "no_goal":
        if ball_goal:
            notes.append("YOLO thấy bóng sau vạch nhưng tỷ số không đổi, giữ kết quả theo scoreboard")
        return ShotOutcomeEvidence(shot_sec, "no_goal", "video_scoreboard", before, score_after, None, ball, tuple(notes))
    if ball_goal:
        return ShotOutcomeEvidence(shot_sec, "goal", "video_ball", before, score_after, None, ball, tuple(notes))
    return ShotOutcomeEvidence(shot_sec, "unknown", None, before, score_after, None, ball, tuple(notes))


def attack_left_to_right(attack: AttackSequenceV2) -> bool | None:
    terminal = next(
        (item for item in attack.possession.actions if item.action_id == attack.terminal.action_id),
        attack.possession.actions[-1],
    )
    return terminal.raw_left_to_right


def infer_attack_outcomes(
    attacks: Sequence[AttackSequenceV2],
    readings: Sequence[ScoreReading],
    frames: Sequence[Any] = (),
    **options: Any,
) -> dict[str, ShotOutcomeEvidence]:
    """Outcome per attack id; score reads after the next attack's shot belong to that shot."""

    shot_times = sorted(attack.terminal.timestamp_sec for attack in attacks)
    outcomes: dict[str, ShotOutcomeEvidence] = {}
    for attack in attacks:
        shot_sec = float(attack.terminal.timestamp_sec)
        after_limit = next((value for value in shot_times if value > shot_sec + 0.5), None)
        ball = ball_in_goal_times(
            frames,
            shot_sec=shot_sec,
            left_to_right=attack_left_to_right(attack),
        )
        outcomes[attack.attack_id] = infer_shot_outcome(
            shot_sec,
            readings,
            after_limit_sec=after_limit,
            ball_times=ball,
            **options,
        )
    return outcomes


def attack_with_video_outcome(
    attack: AttackSequenceV2,
    evidence: ShotOutcomeEvidence,
) -> AttackSequenceV2:
    """Attach the video-estimated outcome to the terminal shot with its provenance."""

    if evidence.outcome == "unknown":
        return attack
    source = evidence.source or "video"
    possession = attack.possession
    actions = actions_with_resolved_outcome(
        possession,
        shot_outcome=evidence.outcome,
        outcome_source=source,
    )
    terminal = replace(possession.terminal, outcome=evidence.outcome)
    return replace(attack, possession=replace(possession, actions=actions, terminal=terminal))
