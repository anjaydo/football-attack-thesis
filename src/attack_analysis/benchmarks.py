from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

from .research import BenchmarkRecord
from .footpass import FOOTPASS_ACTIONS, FOOTPASS_ROLES, normalize_footpass_position
from .xg import heuristic_xg


@dataclass(frozen=True)
class PlayerIdentity:
    match_id: str
    team: str
    jersey: int
    player_name: str
    role: str
    source_id: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PlayerIdentity":
        return cls(
            match_id=str(data["match_id"]),
            team=str(data["team"]),
            jersey=int(data["jersey"]),
            player_name=str(data["player_name"]),
            role=str(data["role"]),
            source_id=str(data["source_id"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "match_id": self.match_id,
            "team": self.team,
            "jersey": self.jersey,
            "player_name": self.player_name,
            "role": self.role,
            "source_id": self.source_id,
        }


class FrozenBenchmarkStore:
    def __init__(
        self,
        records: Iterable[BenchmarkRecord] = (),
        identities: Iterable[PlayerIdentity] = (),
    ) -> None:
        self.records = tuple(records)
        self.identities = tuple(identities)

    @classmethod
    def from_jsonl(
        cls,
        benchmark_path: Path | None = None,
        identity_path: Path | None = None,
    ) -> "FrozenBenchmarkStore":
        records: list[BenchmarkRecord] = []
        identities: list[PlayerIdentity] = []
        if benchmark_path and benchmark_path.exists():
            records = [
                BenchmarkRecord.from_dict(json.loads(line))
                for line in benchmark_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if identity_path and identity_path.exists():
            identities = [
                PlayerIdentity.from_dict(json.loads(line))
                for line in identity_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return cls(records, identities)

    def get_role_benchmark(self, role: str, metric: str | None = None) -> tuple[BenchmarkRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.role == role and (metric is None or record.metric == metric)
        )

    def lookup_player(self, player_name: str) -> tuple[BenchmarkRecord, ...]:
        normalized = player_name.casefold().strip()
        return tuple(
            record
            for record in self.records
            if record.player_name and record.player_name.casefold().strip() == normalized
        )

    def search(self, query: str) -> tuple[BenchmarkRecord, ...]:
        normalized = query.casefold().strip()
        if not normalized:
            return ()
        return tuple(
            record
            for record in self.records
            if normalized
            in " ".join(
                filter(
                    None,
                    (record.player_name, record.team_name, record.role, record.metric, record.season),
                )
            ).casefold()
        )

    def resolve_player_identity(self, match_id: str, team: str, jersey: int) -> PlayerIdentity | None:
        matches = tuple(
            identity
            for identity in self.identities
            if identity.match_id == match_id and identity.team == team and identity.jersey == jersey
        )
        if len(matches) > 1:
            raise ValueError("identity registry contains ambiguous match/team/jersey mapping")
        return matches[0] if matches else None


def build_footpass_role_benchmarks(
    tactical_h5: Path,
    output_path: Path,
    *,
    fps: float = 25.0,
    season: str = "2023-24",
    snapshot_date: str | None = None,
) -> tuple[BenchmarkRecord, ...]:
    try:
        import h5py
        import numpy as np
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'footpass' extra to build role benchmarks") from error
    if fps <= 0:
        raise ValueError("fps must be positive")
    snapshot = snapshot_date or date.today().isoformat()
    aggregates: dict[tuple[str, int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with h5py.File(tactical_h5, "r") as handle:
        for match_key in sorted(handle.keys()):
            rows = handle[match_key][:]
            game_id = match_key.split("_H", 1)[0]
            for player_id in np.unique(rows[:, 1]).astype(int):
                player_rows = rows[rows[:, 1] == player_id]
                role_id = int(round(float(np.median(player_rows[:, 4]))))
                key = (game_id, player_id, role_id)
                minutes = max(0.0, (float(player_rows[:, 0].max()) - float(player_rows[:, 0].min())) / fps / 60.0)
                aggregates[key]["minutes"] += minutes
                for action_id, action_name in FOOTPASS_ACTIONS.items():
                    aggregates[key][action_name] += float((player_rows[:, 13] == action_id).sum())
                for shot_row in player_rows[player_rows[:, 13] == 5]:
                    team = int(shot_row[2])
                    raw_x, raw_y = float(shot_row[5]), float(shot_row[6])
                    x, y = normalize_footpass_position(raw_x, raw_y, team)
                    aggregates[key]["xg"] += heuristic_xg((x, y))

    raw_records: list[tuple[tuple[str, int, int], str, float]] = []
    for key, values in aggregates.items():
        minutes = values["minutes"]
        if minutes < 10.0:
            continue
        for metric in (*FOOTPASS_ACTIONS.values(), "xg"):
            raw_records.append((key, metric, values[metric] * 90.0 / minutes))

    distributions: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (_, _, role_id), metric, value in raw_records:
        distributions[(role_id, metric)].append(value)

    records: list[BenchmarkRecord] = []
    source_url = "https://github.com/JeremieOchin/FOOTPASS"
    for (game_id, player_id, role_id), metric, value in raw_records:
        values = sorted(distributions[(role_id, metric)])
        percentile = 100.0 * sum(item <= value for item in values) / len(values)
        records.append(
            BenchmarkRecord(
                benchmark_id=f"footpass:{game_id}:{player_id}:{metric}",
                player_name=f"anonymous:{game_id}:{player_id}",
                role=FOOTPASS_ROLES.get(role_id, f"role_{role_id}"),
                metric=metric,
                value_per90=round(value, 4),
                percentile=round(percentile, 2),
                season=season,
                sample_size=len(values),
                source_url=source_url,
                snapshot_date=snapshot,
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    return tuple(records)


def write_historical_snapshot(
    records: Iterable[Mapping[str, object]],
    output_path: Path,
    *,
    snapshot_date: str,
    default_season: str,
    default_source_url: str,
) -> tuple[BenchmarkRecord, ...]:
    """Persist a versioned frozen stats snapshot. Live scraped numbers are not used."""

    date.fromisoformat(snapshot_date)
    if not default_season.strip() or not default_source_url.strip():
        raise ValueError("season and source_url must not be empty")
    written: list[BenchmarkRecord] = []
    for index, raw in enumerate(records):
        record = BenchmarkRecord(
            benchmark_id=str(raw.get("benchmark_id") or f"snapshot:{snapshot_date}:{index}"),
            role=str(raw["role"]),
            metric=str(raw["metric"]),
            value_per90=float(raw["value_per90"]),
            percentile=float(raw.get("percentile", 50.0)),
            season=str(raw.get("season") or default_season),
            sample_size=int(raw["sample_size"]),
            source_url=str(raw.get("source_url") or default_source_url),
            snapshot_date=str(raw.get("snapshot_date") or snapshot_date),
            player_name=(str(raw["player_name"]) if raw.get("player_name") else None),
            team_name=(str(raw["team_name"]) if raw.get("team_name") else None),
        )
        record.validate()
        written.append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in written),
        encoding="utf-8",
    )
    return tuple(written)
