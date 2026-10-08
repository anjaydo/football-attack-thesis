"""Versioned registry of FOOTPASS predicted play-by-play files.

Each entry pins one prediction JSON by SHA-256 together with its official
FOOTPASS validation score (``evaluation.py``, tau=0.15, delta=12). Exactly one
entry is ``primary``; Attack-v2 uses it by default. A new file only becomes
primary when its micro-F1 is strictly higher than the current primary. Raw
TAAD output is always an ablation because DST is the thesis decoding path.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import file_sha256

REGISTRY_SCHEMA_VERSION = "prediction-sources/1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "models" / "prediction-sources.json"
DEFAULT_FOOTPASS_ROOT = PROJECT_ROOT.parent / "FOOTPASS"
SOURCE_KINDS = frozenset({"TAAD", "DST"})
SOURCE_ROLES = frozenset({"primary", "ablation"})
OFFICIAL_DELTA = 12
OFFICIAL_CONFIDENCE = 0.15
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MICRO_F1 = re.compile(r"Micro F1:\s*([0-9.]+)")
_OVERALL = re.compile(
    r"^OVERALL\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", re.MULTILINE
)


class PredictionSourceError(ValueError):
    """Registry entry is missing, malformed, or no longer matches its file."""


@dataclass(frozen=True)
class OfficialScore:
    micro_f1: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    ground_truth: int
    delta: int = OFFICIAL_DELTA
    confidence_threshold: float = OFFICIAL_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionSource:
    id: str
    kind: str
    path: str
    sha256: str
    role: str
    official: OfficialScore | None = None
    checkpoint_sha256: str | None = None
    threshold_profile: str | None = None
    registered_at: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not _SOURCE_ID.match(self.id):
            raise PredictionSourceError(f"invalid prediction source id: {self.id!r}")
        if self.kind not in SOURCE_KINDS:
            raise PredictionSourceError(f"{self.id}: kind must be one of {sorted(SOURCE_KINDS)}")
        if self.role not in SOURCE_ROLES:
            raise PredictionSourceError(f"{self.id}: role must be one of {sorted(SOURCE_ROLES)}")
        if self.kind == "TAAD" and self.role == "primary":
            raise PredictionSourceError(f"{self.id}: raw TAAD output cannot be primary")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise PredictionSourceError(f"{self.id}: sha256 must be 64 lowercase hex characters")

    @property
    def micro_f1(self) -> float | None:
        return None if self.official is None else self.official.micro_f1

    @property
    def model_name(self) -> str:
        return self.id

    def resolved_path(self, registry_path: Path) -> Path:
        return _resolve(registry_path, self.path)

    def resolved_threshold_profile(self, registry_path: Path) -> Path | None:
        if not self.threshold_profile:
            return None
        return _resolve(registry_path, self.threshold_profile)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["official"] = None if self.official is None else self.official.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionSource":
        official = payload.get("official")
        return cls(
            id=str(payload["id"]),
            kind=str(payload["kind"]),
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            role=str(payload["role"]),
            official=OfficialScore(**official) if isinstance(official, Mapping) else None,
            checkpoint_sha256=payload.get("checkpoint_sha256"),
            threshold_profile=payload.get("threshold_profile"),
            registered_at=payload.get("registered_at"),
            note=str(payload.get("note") or ""),
        )


@dataclass(frozen=True)
class PredictionRegistry:
    path: Path
    sources: tuple[PredictionSource, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ids = [source.id for source in self.sources]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise PredictionSourceError(f"duplicate prediction source ids: {duplicates}")
        primaries = [source.id for source in self.sources if source.role == "primary"]
        if len(primaries) > 1:
            raise PredictionSourceError(f"more than one primary prediction source: {primaries}")

    @property
    def primary(self) -> PredictionSource | None:
        return next((source for source in self.sources if source.role == "primary"), None)

    def get(self, source_id: str) -> PredictionSource:
        for source in self.sources:
            if source.id == source_id:
                return source
        known = ", ".join(source.id for source in self.sources) or "none"
        raise PredictionSourceError(f"unknown prediction source {source_id!r}; registered: {known}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "primary": None if self.primary is None else self.primary.id,
            "sources": [source.to_dict() for source in self.sources],
        }


def _resolve(registry_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (Path(registry_path).resolve().parent / candidate).resolve()


def _relative_to_registry(registry_path: Path, target: Path) -> str:
    base = Path(registry_path).resolve().parent
    target = Path(target).resolve()
    try:
        return Path(os.path.relpath(target, base)).as_posix()
    except ValueError:
        return target.as_posix()


def load_registry(path: Path | None = None) -> PredictionRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.is_file():
        return PredictionRegistry(path=registry_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise PredictionSourceError(
            f"unsupported prediction registry schema: {payload.get('schema_version')!r}"
        )
    sources = tuple(PredictionSource.from_dict(item) for item in payload.get("sources") or ())
    return PredictionRegistry(path=registry_path, sources=sources)


def save_registry(registry: PredictionRegistry) -> Path:
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return registry.path


def verify_source(registry: PredictionRegistry, source: PredictionSource) -> Path:
    """Return the JSON path after checking the file still matches its pinned hash."""

    path = source.resolved_path(registry.path)
    if not path.is_file():
        raise PredictionSourceError(f"{source.id}: prediction file is missing: {path}")
    actual = file_sha256(path)
    if actual != source.sha256:
        raise PredictionSourceError(
            f"{source.id}: prediction file changed (expected {source.sha256[:12]}, got {actual[:12]})"
        )
    return path


def resolve_prediction_source(
    source_id: str | None = None,
    registry_path: Path | None = None,
) -> tuple[PredictionSource, Path]:
    registry = load_registry(registry_path)
    if source_id is None:
        source = registry.primary
        if source is None:
            raise PredictionSourceError("prediction registry has no primary source")
    else:
        source = registry.get(source_id)
    return source, verify_source(registry, source)


def registry_report(registry_path: Path | None = None) -> dict[str, Any]:
    registry = load_registry(registry_path)
    entries = []
    for source in registry.sources:
        try:
            verify_source(registry, source)
            status = "ok"
        except PredictionSourceError as error:
            status = str(error)
        entries.append({**source.to_dict(), "resolved_path": str(source.resolved_path(registry.path)), "status": status})
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry": str(registry.path),
        "primary": None if registry.primary is None else registry.primary.id,
        "sources": entries,
    }


def parse_official_output(text: str) -> OfficialScore:
    f1_match = _MICRO_F1.search(text)
    overall = _OVERALL.search(text)
    if f1_match is None or overall is None:
        raise PredictionSourceError("FOOTPASS evaluation output has no OVERALL / Micro F1 lines")
    precision, recall, tp, fp, fn, gt = overall.groups()
    return OfficialScore(
        micro_f1=float(f1_match.group(1)),
        precision=float(precision),
        recall=float(recall),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        ground_truth=int(gt),
    )


def official_footpass_score(
    predictions: Path,
    *,
    footpass_root: Path = DEFAULT_FOOTPASS_ROOT,
    ground_truth: Path | None = None,
    python: str = sys.executable,
) -> OfficialScore:
    """Score with FOOTPASS ``evaluation.py`` so numbers match the published 0.719 baseline."""

    root = Path(footpass_root).resolve()
    script = root / "evaluation.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    truth = Path(ground_truth) if ground_truth is not None else root / "playbyplay_GT" / "playbyplay_val.json"
    completed = subprocess.run(
        [
            python,
            str(script),
            "--predictions_file",
            str(Path(predictions).resolve()),
            "--ground_truth_file",
            str(truth.resolve()),
            "--delta",
            str(OFFICIAL_DELTA),
            "--confidence_threshold",
            str(OFFICIAL_CONFIDENCE),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise PredictionSourceError(
            f"FOOTPASS evaluation failed ({completed.returncode}): {completed.stderr.strip()[-400:]}"
        )
    return parse_official_output(completed.stdout)


def decide_role(kind: str, candidate_f1: float, primary: PredictionSource | None) -> str:
    if kind == "TAAD":
        return "ablation"
    if primary is None or primary.micro_f1 is None:
        return "primary"
    return "primary" if candidate_f1 > primary.micro_f1 else "ablation"


def register_prediction_source(
    predictions: Path,
    *,
    source_id: str,
    kind: str,
    score: OfficialScore,
    registry_path: Path | None = None,
    store_directory: Path | None = None,
    checkpoint: Path | None = None,
    threshold_profile: Path | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Pin a prediction JSON and apply the strict primary gate.

    Existing JSON files are never overwritten. A file outside ``store_directory``
    is copied there as ``playbyplay_<id>_val.json``.
    """

    registry = load_registry(registry_path)
    if any(source.id == source_id for source in registry.sources):
        raise PredictionSourceError(f"prediction source {source_id!r} is already registered")
    source_path = Path(predictions).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    digest = file_sha256(source_path)
    stored = source_path
    if store_directory is not None:
        store = Path(store_directory).resolve()
        if source_path.parent != store:
            stored = store / f"playbyplay_{source_id}_val.json"
            if stored.exists():
                if file_sha256(stored) != digest:
                    raise PredictionSourceError(f"refusing to overwrite different file: {stored}")
            else:
                store.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, stored)

    previous_primary = registry.primary
    role = decide_role(kind, score.micro_f1, previous_primary)
    entry = PredictionSource(
        id=source_id,
        kind=kind,
        path=_relative_to_registry(registry.path, stored),
        sha256=digest,
        role=role,
        official=score,
        checkpoint_sha256=None if checkpoint is None else file_sha256(checkpoint),
        threshold_profile=(
            None if threshold_profile is None else _relative_to_registry(registry.path, threshold_profile)
        ),
        registered_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        note=note,
    )
    sources = list(registry.sources)
    if role == "primary" and previous_primary is not None:
        sources = [replace(item, role="ablation") if item.id == previous_primary.id else item for item in sources]
    sources.append(entry)
    updated = PredictionRegistry(path=registry.path, sources=tuple(sources))
    save_registry(updated)

    warnings: list[str] = []
    if role == "primary" and kind == "DST" and previous_primary is not None:
        warnings.append(
            "Class thresholds were fitted on the previous DST file. Refit with "
            "calibrate-footpass-events before using a threshold profile with this source."
        )
    return {
        "registry": str(updated.path),
        "source": entry.to_dict(),
        "role": role,
        "previous_primary": None if previous_primary is None else previous_primary.id,
        "previous_primary_micro_f1": None if previous_primary is None else previous_primary.micro_f1,
        "promoted": role == "primary",
        "warnings": warnings,
    }


def source_choices(registry_path: Path | None = None) -> Sequence[PredictionSource]:
    registry = load_registry(registry_path)
    return sorted(registry.sources, key=lambda item: (item.role != "primary", item.id))
