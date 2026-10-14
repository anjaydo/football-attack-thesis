"""Goal times per FOOTPASS match from the broadcast scoreboard, mapped onto attacks.

FOOTPASS has no shot outcome.  The score bug in the match video does: every
stable score increase is a goal.  The scoring attempt is the last live attempt
before the score changes, preferably the one followed by the celebration
stoppage.  Feeds without a readable score bug fall back to the event-only
restart pattern (``restart_goal_times``), which is weaker evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .footpass import split_footpass_match_key
from .models import ObservedActionV2, PossessionSequenceV2

SCOREBOARD_GOALS_SCHEMA = "footpass-scoreboard-goals/1.0"
DEFAULT_SCOREBOARD_GOALS = Path("data") / "footpass-scoreboard-goals.json"
SHOT_ACTION_TYPES = frozenset({"shot", "shot_attempt", "headed_attempt", "headed_shot"})
SCORING_LOOKBACK_SEC = 150.0
CELEBRATION_GAP_SEC = 15.0
RESTART_MIN_STOPPAGE_SEC = 45.0
RESTART_X_RANGE = (20.0, 60.0)


def stable_score_changes(
    readings: Sequence[Mapping[str, object]],
    *,
    min_agreeing: int = 3,
) -> list[dict[str, object]]:
    """Score states held for ``min_agreeing`` consecutive reads; the first one is ``initial``."""

    changes: list[dict[str, object]] = []
    current: tuple[int, int] | None = None
    pending: tuple[int, int] | None = None
    pending_count = 0
    pending_first = 0.0
    for item in readings:
        score = (int(item["home"]), int(item["away"]))
        if current is not None and score == current:
            pending, pending_count = None, 0
            continue
        if pending == score:
            pending_count += 1
        else:
            pending, pending_count, pending_first = score, 1, float(item["timestamp_sec"])
        if pending_count < min_agreeing:
            continue
        changes.append(
            {
                "timestamp_sec": pending_first,
                "score": list(score),
                "previous": None if current is None else list(current),
            }
        )
        current = score
        pending, pending_count = None, 0
    return changes


def goal_times_from_changes(changes: Sequence[Mapping[str, object]]) -> list[float]:
    """Score increases by one that are never reverted later (OCR flips are dropped)."""

    accepted: list[float] = []
    history: list[tuple[int, int]] = []
    for change in changes:
        score = tuple(int(value) for value in change["score"])  # type: ignore[union-attr]
        if not history:
            history = [score]
            continue
        if len(history) >= 2 and score == history[-2]:
            history.pop()
            if accepted:
                accepted.pop()
            continue
        previous = history[-1]
        if sum(score) == sum(previous) + 1 and score[0] >= previous[0] and score[1] >= previous[1]:
            accepted.append(float(change["timestamp_sec"]))  # type: ignore[arg-type]
            history.append(score)
        elif sum(score) > sum(previous):
            history.append(score)
    return accepted


def goal_times_from_readings(readings: Sequence[Mapping[str, object]]) -> list[float]:
    return goal_times_from_changes(stable_score_changes(readings))


def restart_goal_times(
    actions: Sequence[ObservedActionV2],
    *,
    min_stoppage_sec: float = RESTART_MIN_STOPPAGE_SEC,
    restart_x: tuple[float, float] = RESTART_X_RANGE,
) -> list[float]:
    """Attempts followed by a celebration-length stoppage and a restart by the other team in its own half.

    FOOTPASS rarely labels the kick-off pass itself (annotation resumes when
    the camera returns), so the restart is recognised by who plays first and
    where.  Goal kicks start deeper and corners belong to the attacking team.
    Against 162 scoreboard goals: recall 0.73, precision 0.94.
    """

    live = [item for item in actions if not item.replay]
    times: list[float] = []
    for index, action in enumerate(live):
        if not is_attempt_action(action) or not action.actor_team_id:
            continue
        restart = [item for item in live[index + 1 :] if item.timestamp_sec > action.timestamp_sec + 0.5][:2]
        if not restart or restart[0].timestamp_sec - action.timestamp_sec < min_stoppage_sec:
            continue
        if any(item.actor_team_id == action.actor_team_id for item in restart):
            continue
        first = restart[0]
        if first.x is None or not restart_x[0] <= first.x <= restart_x[1]:
            continue
        times.append(action.timestamp_sec + 1.0)
    return times


def is_attempt_action(action: ObservedActionV2) -> bool:
    """Shot, promoted headed attempt, or a header inside the attacked penalty box."""

    if action.action_type in SHOT_ACTION_TYPES or "headed_attempt" in action.qualifiers:
        return True
    return (
        action.action_type == "header"
        and action.x is not None
        and action.y is not None
        and action.x >= 88.5
        and 13.84 <= action.y <= 54.16
    )


def scoring_action(
    actions: Sequence[ObservedActionV2],
    goal_sec: float,
    *,
    lookback_sec: float = SCORING_LOOKBACK_SEC,
) -> ObservedActionV2 | None:
    """Last live attempt before ``goal_sec``, preferring one followed by the celebration stoppage."""

    live = [item for item in actions if not item.replay and item.timestamp_sec <= goal_sec]
    candidates = [
        item
        for item in live
        if is_attempt_action(item) and item.timestamp_sec >= goal_sec - lookback_sec
    ]
    if not candidates:
        return None
    for item in reversed(candidates):
        later = next((other for other in live if other.timestamp_sec > item.timestamp_sec + 0.5), None)
        if later is None or later.timestamp_sec - item.timestamp_sec >= CELEBRATION_GAP_SEC:
            return item
    return candidates[-1]


@dataclass(frozen=True)
class MatchGoalTruth:
    match_id: str
    goal_times_sec: tuple[float, ...]
    source: str
    final_score: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_times_sec": list(self.goal_times_sec),
            "source": self.source,
            "final_score": None if self.final_score is None else list(self.final_score),
        }


@dataclass(frozen=True)
class ScoreboardGoalIndex:
    matches: Mapping[str, MatchGoalTruth]

    @classmethod
    def load(cls, path: Path) -> "ScoreboardGoalIndex":
        if not path.is_file():
            return cls({})
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCOREBOARD_GOALS_SCHEMA:
            raise ValueError(f"{path} is not {SCOREBOARD_GOALS_SCHEMA}")
        matches: dict[str, MatchGoalTruth] = {}
        for match_id, item in (payload.get("matches") or {}).items():
            final = item.get("final_score")
            matches[str(match_id)] = MatchGoalTruth(
                match_id=str(match_id),
                goal_times_sec=tuple(float(value) for value in item.get("goal_times_sec") or ()),
                source=str(item.get("source") or "video_scoreboard"),
                final_score=None if final is None else (int(final[0]), int(final[1])),
            )
        return cls(matches)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCOREBOARD_GOALS_SCHEMA,
            "matches": {
                key: self.matches[key].to_dict()
                for key in sorted(self.matches, key=lambda value: int(value.split("_")[-1]))
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    def for_match_key(self, match_key: str) -> MatchGoalTruth | None:
        match_id, _period = split_footpass_match_key(match_key)
        return self.matches.get(match_id)


def match_truth_from_readings(
    match_id: str,
    readings: Sequence[Mapping[str, object]],
) -> MatchGoalTruth | None:
    changes = stable_score_changes(readings)
    if not changes:
        return None
    final = tuple(int(value) for value in changes[-1]["score"]) if changes else None  # type: ignore[union-attr]
    return MatchGoalTruth(
        match_id=match_id,
        goal_times_sec=tuple(goal_times_from_changes(changes)),
        source="video_scoreboard",
        final_score=final,  # type: ignore[arg-type]
    )


def match_truth_from_restarts(
    match_id: str,
    halves: Iterable[Sequence[ObservedActionV2]],
) -> MatchGoalTruth:
    times = sorted(value for actions in halves for value in restart_goal_times(actions))
    return MatchGoalTruth(match_id=match_id, goal_times_sec=tuple(times), source="event_restart")


def read_scoreboard_readings(
    match_id: str,
    readings_dir: Path,
    video_root: Path | None = None,
) -> list[dict[str, object]] | None:
    """Cached ``<match>.readings.json``; scans ``<video_root>/<match>.mp4`` when the cache is missing."""

    cache = readings_dir / f"{match_id}.readings.json"
    if cache.is_file():
        return list(json.loads(cache.read_text(encoding="utf-8")).get("readings") or [])
    video = None if video_root is None else video_root / f"{match_id}.mp4"
    if video is None or not video.is_file():
        return None
    from .shot_outcome import ScoreboardReader

    readings_dir.mkdir(parents=True, exist_ok=True)
    return [item.to_dict() for item in ScoreboardReader(step_sec=2.0).read_video(video, cache_path=cache)]


def possession_outcomes(
    match_key: str,
    observed_actions: Sequence[ObservedActionV2],
    possessions: Iterable[PossessionSequenceV2],
    index: ScoreboardGoalIndex,
) -> dict[str, tuple[str, str]]:
    """``possession_id -> (outcome, source)`` for attempt-ending possessions of one half.

    The possession holding a goal's scoring attempt is ``goal``.  With a
    scoreboard source every other attempt-ending possession is ``no_goal``;
    restart evidence only marks goals.
    """

    truth = index.for_match_key(match_key)
    if truth is None or not observed_actions:
        return {}
    possessions = tuple(possessions)
    by_action = {action.action_id: item for item in possessions for action in item.actions}
    start = observed_actions[0].timestamp_sec
    end = observed_actions[-1].timestamp_sec
    goals = [value for value in truth.goal_times_sec if start <= value <= end + SCORING_LOOKBACK_SEC]
    outcomes: dict[str, tuple[str, str]] = {}
    for goal_sec in goals:
        action = scoring_action(observed_actions, goal_sec)
        possession = by_action.get(action.action_id) if action is not None else None
        if possession is not None:
            outcomes[possession.possession_id] = ("goal", truth.source)
    if truth.source == "video_scoreboard":
        for item in possessions:
            if item.possession_id in outcomes or item.terminal is None:
                continue
            if any(is_attempt_action(action) for action in item.actions):
                outcomes[item.possession_id] = ("no_goal", truth.source)
    return outcomes


__all__ = [
    "DEFAULT_SCOREBOARD_GOALS",
    "MatchGoalTruth",
    "SCOREBOARD_GOALS_SCHEMA",
    "SCORING_LOOKBACK_SEC",
    "ScoreboardGoalIndex",
    "goal_times_from_changes",
    "goal_times_from_readings",
    "is_attempt_action",
    "match_truth_from_readings",
    "match_truth_from_restarts",
    "possession_outcomes",
    "read_scoreboard_readings",
    "restart_goal_times",
    "scoring_action",
    "stable_score_changes",
]
