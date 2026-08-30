"""Immutable release and one-shot final-test controls for Attack-v2.

The controls in this module are intentionally stricter than an experiment
tracker.  A frozen release binds the development-selected operating point to
the exact config, model, schema, rubric, and development labels.  Claiming the
final test is an irreversible action: even a failed evaluation process leaves
the release consumed, preventing test-set feedback from becoming a tuning
loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .action_spotting_v2 import (
    ActionSpottingEvaluationProtocol,
    FrozenOperatingPoint,
    OperatingPointMeasurement,
    select_and_freeze_attack_operating_point,
)
from .provenance import canonical_json, content_sha256, file_sha256, tree_sha256


RELEASE_SCHEMA_VERSION = "attack-v2-frozen-release/v2"
FINAL_TEST_CLAIM_SCHEMA_VERSION = "attack-v2-final-test-claim/v1"
FINAL_TEST_RESULT_SCHEMA_VERSION = "attack-v2-final-test-result/v1"
FINAL_TEST_EVALUATION_PROTOCOL: dict[str, object] = {
    "schema_version": "attack-sequence-evaluation/v2",
    "terminal_compatibility": "exact",
    "tiou_thresholds": [0.25, 0.5, 0.75],
    "primary_tiou_threshold": 0.5,
    "bootstrap_samples": 2000,
    "bootstrap_seed": 42,
    "abstention_policy": "missed_prediction",
    "frozen_baseline_required": True,
    "improvement_gate": "paired_match_cluster_bootstrap_f1_ci_lower_gt_0",
}

_SHA256_LENGTH = 64


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    object_name: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ReleaseIntegrityError(f"invalid {object_name} fields: {', '.join(details)}")


class ReleaseIntegrityError(ValueError):
    """Raised when a frozen release or ledger record fails self-verification."""


class FinalTestAlreadyClaimedError(PermissionError):
    """Raised when a release has already consumed its sole final-test run."""


def _require_sha256(value: object, field_name: str) -> str:
    digest = str(value)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def path_sha256(path: str | Path) -> str:
    """Hash one release input, supporting either a file or a model bundle."""

    candidate = Path(path)
    if candidate.is_file():
        return file_sha256(candidate)
    if candidate.is_dir():
        return tree_sha256(candidate)
    raise FileNotFoundError(candidate)


def final_test_evaluation_protocol_sha256(
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
) -> str:
    """Return the sole permitted final-test protocol hash.

    Development experiments may use fewer bootstrap samples, but a frozen
    release cannot alter the declared 2,000-resample, seed-42 final protocol.
    """

    if bootstrap_samples != 2000 or bootstrap_seed != 42:
        raise ValueError(
            "frozen final-test evaluation requires 2,000 bootstrap samples and seed 42"
        )
    return content_sha256(FINAL_TEST_EVALUATION_PROTOCOL)


@dataclass(frozen=True)
class ReleaseComponentHashes:
    config_sha256: str
    model_sha256: str
    schema_sha256: str
    rubric_sha256: str
    development_dataset_sha256: str
    frozen_baseline_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in self.to_dict().items():
            _require_sha256(value, field_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "config_sha256": self.config_sha256,
            "model_sha256": self.model_sha256,
            "schema_sha256": self.schema_sha256,
            "rubric_sha256": self.rubric_sha256,
            "development_dataset_sha256": self.development_dataset_sha256,
            "frozen_baseline_sha256": self.frozen_baseline_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReleaseComponentHashes":
        required = {
            "config_sha256",
            "model_sha256",
            "schema_sha256",
            "rubric_sha256",
            "development_dataset_sha256",
            "frozen_baseline_sha256",
        }
        _require_exact_keys(payload, required, "release component hash")
        try:
            return cls(**{name: str(payload[name]) for name in required})
        except KeyError as error:
            raise ReleaseIntegrityError(
                f"release component hash is missing {error.args[0]}"
            ) from error


def _operating_point_hash_payload(payload: Mapping[str, object]) -> dict[str, object]:
    required = (
        "release_id",
        "development_dataset_hash",
        "selected",
        "evaluated_count",
        "protocol",
        "selection_rule",
    )
    try:
        return {name: payload[name] for name in required}
    except KeyError as error:
        raise ReleaseIntegrityError(
            f"frozen operating point is missing {error.args[0]}"
        ) from error


def _verify_operating_point(
    payload: Mapping[str, object],
    *,
    release_id: str,
    development_dataset_sha256: str,
) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "release_id",
            "development_dataset_hash",
            "selected",
            "evaluated_count",
            "protocol",
            "selection_rule",
            "freeze_hash",
            "frozen",
        },
        "frozen operating point",
    )
    if payload.get("schema_version") != "action-operating-point-v2.0":
        raise ReleaseIntegrityError("unsupported frozen operating-point schema")
    if payload.get("frozen") is not True:
        raise ReleaseIntegrityError("operating point is not frozen")
    if payload.get("release_id") != release_id:
        raise ReleaseIntegrityError("operating-point release_id mismatch")
    if payload.get("development_dataset_hash") != development_dataset_sha256:
        raise ReleaseIntegrityError("operating-point development dataset mismatch")

    selected = payload.get("selected")
    protocol = payload.get("protocol")
    if not isinstance(selected, Mapping) or not isinstance(protocol, Mapping):
        raise ReleaseIntegrityError("invalid frozen operating-point payload")
    _require_exact_keys(
        selected,
        {"threshold", "attack_precision", "attack_recall", "label", "attack_f1"},
        "selected operating point",
    )
    _require_exact_keys(
        protocol,
        {"name", "tolerance_frames", "tuned_split", "primary_selection_metric"},
        "operating-point protocol",
    )
    try:
        precision = float(selected["attack_precision"])
        recall = float(selected["attack_recall"])
        threshold = float(selected["threshold"])
        stored_f1 = float(selected["attack_f1"])
        evaluated_count = int(payload["evaluated_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseIntegrityError("invalid frozen operating-point values") from error
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (
        precision,
        recall,
        threshold,
        stored_f1,
    )):
        raise ReleaseIntegrityError("operating-point metrics must be finite in [0, 1]")
    expected_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if not math.isclose(stored_f1, expected_f1, rel_tol=1e-12, abs_tol=1e-12):
        raise ReleaseIntegrityError("stored operating-point F1 is inconsistent")
    if evaluated_count < 1:
        raise ReleaseIntegrityError("operating-point candidate count must be positive")
    if protocol.get("tuned_split") != "development":
        raise ReleaseIntegrityError("operating point was not selected on development")
    if protocol.get("primary_selection_metric") != "attack_f1_tiou_0.5":
        raise ReleaseIntegrityError("unexpected operating-point selection metric")
    if int(protocol.get("tolerance_frames", -1)) != 12:
        raise ReleaseIntegrityError("official action tolerance must remain ±12 frames")

    expected_hash = content_sha256(_operating_point_hash_payload(payload))
    if payload.get("freeze_hash") != expected_hash:
        raise ReleaseIntegrityError("frozen operating-point hash mismatch")


@dataclass(frozen=True)
class FrozenAttackV2Release:
    release_id: str
    component_hashes: ReleaseComponentHashes
    operating_point: Mapping[str, object]
    development_sweep_sha256: str
    release_sha256: str
    frozen: bool = True
    schema_version: str = RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.release_id or self.release_id.strip() != self.release_id:
            raise ValueError("release_id must be a non-empty trimmed string")
        _require_sha256(self.development_sweep_sha256, "development_sweep_sha256")
        _require_sha256(self.release_sha256, "release_sha256")
        if self.schema_version != RELEASE_SCHEMA_VERSION or not self.frozen:
            raise ValueError("release must use the frozen Attack-v2 release schema")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "state": "frozen",
            "component_hashes": self.component_hashes.to_dict(),
            "development_selection": {
                "sweep_sha256": self.development_sweep_sha256,
                "operating_point": dict(self.operating_point),
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "release_sha256": self.release_sha256,
        }

    def verify(self) -> None:
        _verify_operating_point(
            self.operating_point,
            release_id=self.release_id,
            development_dataset_sha256=self.component_hashes.development_dataset_sha256,
        )
        expected = content_sha256(self.identity_payload())
        if self.release_sha256 != expected:
            raise ReleaseIntegrityError("frozen release manifest hash mismatch")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenAttackV2Release":
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "release_id",
                "state",
                "component_hashes",
                "development_selection",
                "release_sha256",
            },
            "frozen release manifest",
        )
        if payload.get("schema_version") != RELEASE_SCHEMA_VERSION:
            raise ReleaseIntegrityError("unsupported Attack-v2 release schema")
        if payload.get("state") != "frozen":
            raise ReleaseIntegrityError("release manifest is not frozen")
        components = payload.get("component_hashes")
        selection = payload.get("development_selection")
        if not isinstance(components, Mapping) or not isinstance(selection, Mapping):
            raise ReleaseIntegrityError("release manifest is missing frozen components")
        _require_exact_keys(
            selection,
            {"sweep_sha256", "operating_point"},
            "development selection",
        )
        operating_point = selection.get("operating_point")
        if not isinstance(operating_point, Mapping):
            raise ReleaseIntegrityError("release manifest is missing its operating point")
        try:
            record = cls(
                release_id=str(payload["release_id"]),
                component_hashes=ReleaseComponentHashes.from_dict(components),
                operating_point=dict(operating_point),
                development_sweep_sha256=str(selection["sweep_sha256"]),
                release_sha256=str(payload["release_sha256"]),
            )
        except KeyError as error:
            raise ReleaseIntegrityError(
                f"release manifest is missing {error.args[0]}"
            ) from error
        record.verify()
        return record


def freeze_attack_v2_release(
    *,
    release_id: str,
    component_hashes: ReleaseComponentHashes,
    measurements: Iterable[OperatingPointMeasurement],
    split: str = "development",
    protocol: ActionSpottingEvaluationProtocol | None = None,
) -> FrozenAttackV2Release:
    """Select and bind the only legal operating point for one release."""

    candidates = tuple(measurements)
    operating_point: FrozenOperatingPoint = select_and_freeze_attack_operating_point(
        candidates,
        split=split,
        release_id=release_id,
        development_dataset_hash=component_hashes.development_dataset_sha256,
        protocol=protocol,
    )
    normalized_sweep = [
        {
            "threshold": item.threshold,
            "attack_precision": item.attack_precision,
            "attack_recall": item.attack_recall,
            "attack_f1": item.attack_f1,
            "label": item.label,
        }
        for item in candidates
    ]
    operating_payload = operating_point.to_dict()
    draft = FrozenAttackV2Release(
        release_id=release_id,
        component_hashes=component_hashes,
        operating_point=operating_payload,
        development_sweep_sha256=content_sha256(normalized_sweep),
        release_sha256="0" * _SHA256_LENGTH,
    )
    result = FrozenAttackV2Release(
        release_id=draft.release_id,
        component_hashes=draft.component_hashes,
        operating_point=draft.operating_point,
        development_sweep_sha256=draft.development_sweep_sha256,
        release_sha256=content_sha256(draft.identity_payload()),
    )
    result.verify()
    return result


def write_frozen_release_manifest(record: FrozenAttackV2Release, path: str | Path) -> None:
    """Create a release manifest once and refuse any changed replacement."""

    record.verify()
    destination = Path(path)
    serialized = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") == serialized:
            return
        raise FileExistsError(f"refusing to replace frozen release manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace frozen release manifest: {destination}"
        ) from error


def load_frozen_release_manifest(path: str | Path) -> FrozenAttackV2Release:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReleaseIntegrityError("release manifest must be a JSON object")
    return FrozenAttackV2Release.from_dict(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ledger_key(release_id: str) -> str:
    return hashlib.sha256(release_id.encode("utf-8")).hexdigest()


def _exclusive_json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_self_hash(
    payload: Mapping[str, object],
    *,
    hash_field: str,
    expected_schema: str,
) -> None:
    if payload.get("schema_version") != expected_schema:
        raise ReleaseIntegrityError("unsupported final-test ledger schema")
    supplied = _require_sha256(payload.get(hash_field), hash_field)
    identity = {key: value for key, value in payload.items() if key != hash_field}
    if supplied != content_sha256(identity):
        raise ReleaseIntegrityError("final-test ledger record hash mismatch")


def claim_final_test_once(
    release: FrozenAttackV2Release,
    ledger_directory: str | Path,
    *,
    final_test_labels_sha256: str,
    predictions_sha256: str,
    evaluation_protocol_sha256: str,
) -> dict[str, object]:
    """Irreversibly consume the final test for ``release`` before labels are read."""

    release.verify()
    labels_hash = _require_sha256(final_test_labels_sha256, "final_test_labels_sha256")
    prediction_hash = _require_sha256(predictions_sha256, "predictions_sha256")
    protocol_hash = _require_sha256(
        evaluation_protocol_sha256, "evaluation_protocol_sha256"
    )
    key = _ledger_key(release.release_id)
    claim_path = Path(ledger_directory) / "claims" / f"{key}.json"
    identity: dict[str, object] = {
        "schema_version": FINAL_TEST_CLAIM_SCHEMA_VERSION,
        "state": "claimed",
        "release_id": release.release_id,
        "release_sha256": release.release_sha256,
        "final_test_labels_sha256": labels_hash,
        "predictions_sha256": prediction_hash,
        "evaluation_protocol_sha256": protocol_hash,
        "claimed_at_utc": _utc_now(),
    }
    record = {**identity, "claim_sha256": content_sha256(identity)}
    try:
        _exclusive_json_write(claim_path, record)
    except FileExistsError as error:
        try:
            existing = json.loads(claim_path.read_text(encoding="utf-8"))
            if isinstance(existing, Mapping):
                _verify_self_hash(
                    existing,
                    hash_field="claim_sha256",
                    expected_schema=FINAL_TEST_CLAIM_SCHEMA_VERSION,
                )
        except (OSError, ValueError, json.JSONDecodeError) as integrity_error:
            raise ReleaseIntegrityError(
                "the final-test claim exists but is corrupt; the release remains consumed"
            ) from integrity_error
        raise FinalTestAlreadyClaimedError(
            f"final test already claimed for release {release.release_id}"
        ) from error
    return {**record, "claim_path": str(claim_path)}


def complete_final_test_claim(
    release: FrozenAttackV2Release,
    ledger_directory: str | Path,
    *,
    claim_sha256: str,
    result_sha256: str,
) -> dict[str, object]:
    """Append an immutable completion record; never changes the original claim."""

    release.verify()
    claim_hash = _require_sha256(claim_sha256, "claim_sha256")
    result_hash = _require_sha256(result_sha256, "result_sha256")
    key = _ledger_key(release.release_id)
    claim_path = Path(ledger_directory) / "claims" / f"{key}.json"
    if not claim_path.is_file():
        raise PermissionError("cannot complete a final-test evaluation without a claim")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if not isinstance(claim, Mapping):
        raise ReleaseIntegrityError("invalid final-test claim record")
    _verify_self_hash(
        claim,
        hash_field="claim_sha256",
        expected_schema=FINAL_TEST_CLAIM_SCHEMA_VERSION,
    )
    if claim.get("release_sha256") != release.release_sha256 or claim.get(
        "claim_sha256"
    ) != claim_hash:
        raise ReleaseIntegrityError("completion does not match the immutable claim")

    result_path = Path(ledger_directory) / "results" / f"{key}.json"
    identity: dict[str, object] = {
        "schema_version": FINAL_TEST_RESULT_SCHEMA_VERSION,
        "state": "completed",
        "release_id": release.release_id,
        "release_sha256": release.release_sha256,
        "claim_sha256": claim_hash,
        "result_sha256": result_hash,
        "completed_at_utc": _utc_now(),
    }
    record = {**identity, "completion_sha256": content_sha256(identity)}
    try:
        _exclusive_json_write(result_path, record)
    except FileExistsError as error:
        raise FinalTestAlreadyClaimedError(
            f"final-test result already recorded for release {release.release_id}"
        ) from error
    return {**record, "result_path": str(result_path)}


def final_test_ledger_status(
    release: FrozenAttackV2Release,
    ledger_directory: str | Path,
) -> dict[str, object]:
    """Read and verify claim/completion state without changing the ledger."""

    release.verify()
    key = _ledger_key(release.release_id)
    root = Path(ledger_directory)
    claim_path = root / "claims" / f"{key}.json"
    result_path = root / "results" / f"{key}.json"
    if not claim_path.exists():
        return {
            "release_id": release.release_id,
            "release_sha256": release.release_sha256,
            "state": "available",
        }
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if not isinstance(claim, Mapping):
        raise ReleaseIntegrityError("invalid final-test claim record")
    _verify_self_hash(
        claim,
        hash_field="claim_sha256",
        expected_schema=FINAL_TEST_CLAIM_SCHEMA_VERSION,
    )
    if claim.get("release_sha256") != release.release_sha256:
        raise ReleaseIntegrityError("ledger release hash does not match frozen manifest")
    status: dict[str, object] = {**dict(claim), "state": "claimed"}
    if result_path.exists():
        completion = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(completion, Mapping):
            raise ReleaseIntegrityError("invalid final-test completion record")
        _verify_self_hash(
            completion,
            hash_field="completion_sha256",
            expected_schema=FINAL_TEST_RESULT_SCHEMA_VERSION,
        )
        if completion.get("claim_sha256") != claim.get("claim_sha256"):
            raise ReleaseIntegrityError("completion is not linked to the final-test claim")
        status = {**dict(claim), "state": "completed", "completion": dict(completion)}
    return status


__all__ = [
    "FINAL_TEST_CLAIM_SCHEMA_VERSION",
    "FINAL_TEST_EVALUATION_PROTOCOL",
    "FINAL_TEST_RESULT_SCHEMA_VERSION",
    "RELEASE_SCHEMA_VERSION",
    "FinalTestAlreadyClaimedError",
    "FrozenAttackV2Release",
    "ReleaseComponentHashes",
    "ReleaseIntegrityError",
    "claim_final_test_once",
    "complete_final_test_claim",
    "final_test_ledger_status",
    "final_test_evaluation_protocol_sha256",
    "freeze_attack_v2_release",
    "load_frozen_release_manifest",
    "path_sha256",
    "write_frozen_release_manifest",
]
