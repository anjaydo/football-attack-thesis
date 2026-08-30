from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


MANIFEST_VERSION = "attack-analysis-manifest-v2"
PROVENANCE_SCHEMA_VERSION = "artifact-provenance-v2"

SourceTier = Literal[
    "oracle_actions_oracle_tactical",
    "predicted_actions_oracle_tactical",
    "video_only",
    "fused",
]

SOURCE_TIERS: frozenset[str] = frozenset(
    {
        "oracle_actions_oracle_tactical",
        "predicted_actions_oracle_tactical",
        "video_only",
        "fused",
    }
)

_SECRET_KEY_PARTS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)
_SECRET_FILE_NAMES = frozenset(
    {
        ".env",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
    }
)
_TREE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
)


class StaleArtifactError(ValueError):
    """Raised when a persisted artifact does not match required provenance."""


@dataclass(frozen=True)
class ManifestComparison:
    status: Literal["current", "stale", "legacy_unvalidated"]
    mismatches: tuple[str, ...] = ()

    @property
    def is_current(self) -> bool:
        return self.status == "current"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "is_current": self.is_current,
            "mismatches": list(self.mismatches),
        }


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def is_secret_key(value: object) -> bool:
    key = _normalized_key(value)
    return key in _SECRET_KEY_PARTS or any(
        key.endswith(f"_{part}") for part in _SECRET_KEY_PARTS
    )


def _is_secret_path(path: Path) -> bool:
    name = path.name.casefold()
    normalized_stem = _normalized_key(path.stem)
    return (
        name in _SECRET_FILE_NAMES
        or path.suffix.casefold() in {".key", ".pem", ".p12", ".pfx"}
        or is_secret_key(normalized_stem)
    )


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provenance values must not contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, bytes):
        # Binary values are represented by content digest, never embedded in a manifest.
        return {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if is_secret_key(key):
                continue
            result[str(key)] = _canonicalize(value[key])
        return result
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically while dropping secret-bearing fields."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path)
    if _is_secret_path(file_path):
        raise ValueError(f"refusing to fingerprint secret-bearing file: {file_path.name}")
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def files_sha256(paths: Iterable[str | Path]) -> str:
    """Hash a deterministic file set without exposing absolute paths or contents."""

    records: list[dict[str, object]] = []
    for path in (Path(item) for item in paths):
        records.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return content_sha256(sorted(records, key=canonical_json))


def tree_sha256(
    root: str | Path,
    *,
    suffixes: Iterable[str] | None = None,
) -> str:
    """Hash relative names and contents of a deterministic, non-secret file tree."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    allowed_suffixes = (
        frozenset(str(suffix).casefold() for suffix in suffixes)
        if suffixes is not None
        else None
    )
    records: list[dict[str, object]] = []
    for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root_path)
        if any(part in _TREE_EXCLUDED_PARTS for part in relative.parts):
            continue
        if not path.is_file() or path.is_symlink() or _is_secret_path(path):
            continue
        if allowed_suffixes is not None and path.suffix.casefold() not in allowed_suffixes:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return content_sha256(records)


def validate_source_tier(source_tier: str) -> SourceTier:
    if source_tier not in SOURCE_TIERS:
        allowed = ", ".join(sorted(SOURCE_TIERS))
        raise ValueError(f"source_tier must be one of: {allowed}")
    return source_tier  # type: ignore[return-value]


def build_provenance(
    *,
    source_tier: SourceTier,
    video_sha256: str,
    config: Any,
    model_fingerprints: Any,
    schema_versions: Any,
    postprocess: Any = None,
    dataset_paths: Iterable[str | Path] = (),
    content_root: str | Path | None = None,
) -> dict[str, str]:
    """Build the mandatory, hash-only provenance contract for an artifact."""

    validate_source_tier(source_tier)
    dataset_digest = files_sha256(dataset_paths)
    tree_digest = (
        tree_sha256(content_root, suffixes={".py", ".toml", ".json", ".yaml", ".yml"})
        if content_root is not None
        else content_sha256([])
    )
    provenance = {
        "provenance_schema": PROVENANCE_SCHEMA_VERSION,
        "video_sha256": str(video_sha256),
        "dataset_sha256": dataset_digest,
        "config_sha256": content_sha256(config),
        "model_sha256": content_sha256(model_fingerprints),
        "schema_sha256": content_sha256(schema_versions),
        "postprocess_sha256": content_sha256(postprocess),
        "content_tree_sha256": tree_digest,
    }
    provenance["artifact_identity_sha256"] = content_sha256(
        {
            "manifest_version": MANIFEST_VERSION,
            "source_tier": source_tier,
            **provenance,
        }
    )
    return provenance


def mark_legacy_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with an explicit non-validated state for pre-v2 manifests."""

    result = dict(manifest)
    if result.get("manifest_version") != MANIFEST_VERSION or not isinstance(
        result.get("provenance"), Mapping
    ):
        result["provenance_status"] = "legacy_unvalidated"
        result["legacy_unvalidated"] = True
    return result


def compare_manifest_provenance(
    manifest: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
    source_tier: SourceTier,
) -> ManifestComparison:
    """Compare all mandatory fields; legacy and partial manifests never pass."""

    validate_source_tier(source_tier)
    declared_status = manifest.get("provenance_status")
    if (
        declared_status == "legacy_unvalidated"
        or manifest.get("validation_status") == "legacy_unvalidated"
        or manifest.get("legacy_unvalidated") is True
    ):
        return ManifestComparison("legacy_unvalidated", ("provenance_status",))
    if declared_status == "stale":
        return ManifestComparison("stale", ("provenance_status",))
    if manifest.get("manifest_version") != MANIFEST_VERSION or not isinstance(
        manifest.get("provenance"), Mapping
    ):
        return ManifestComparison("legacy_unvalidated", ("manifest_version", "provenance"))

    actual = manifest["provenance"]
    mismatches: list[str] = []
    if manifest.get("source_tier") != source_tier:
        mismatches.append("source_tier")
    for key, expected in expected_provenance.items():
        if actual.get(key) != expected:
            mismatches.append(f"provenance.{key}")
    unexpected_schema = actual.get("provenance_schema") != PROVENANCE_SCHEMA_VERSION
    if unexpected_schema and "provenance.provenance_schema" not in mismatches:
        mismatches.append("provenance.provenance_schema")
    return ManifestComparison("stale", tuple(mismatches)) if mismatches else ManifestComparison("current")


def require_current_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
    source_tier: SourceTier,
) -> None:
    comparison = compare_manifest_provenance(
        manifest,
        expected_provenance=expected_provenance,
        source_tier=source_tier,
    )
    if not comparison.is_current:
        details = ", ".join(comparison.mismatches) or comparison.status
        raise StaleArtifactError(f"artifact provenance is {comparison.status}: {details}")


def invalidate_artifact_tree(
    root: str | Path,
    *,
    reason: str,
    replacement: str | None = None,
) -> dict[str, object]:
    """Mark a persisted artifact tree stale without deleting user evidence.

    Every readable ``manifest.json`` is updated in place and a root marker
    records the exact pre-invalidation manifest hashes.  Standalone legacy
    files remain available for audit, but the marker makes the directory
    explicitly non-current.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    if root_path == Path(root_path.anchor):
        raise ValueError("refusing to invalidate a filesystem root")
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("stale-artifact reason must not be empty")
    manifests = sorted(root_path.rglob("manifest.json"), key=lambda item: item.as_posix())
    before = [
        {
            "path": path.relative_to(root_path).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in manifests
        if path.is_file() and not path.is_symlink()
    ]
    updated: list[str] = []
    unreadable: list[str] = []
    for path in manifests:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unreadable.append(path.relative_to(root_path).as_posix())
            continue
        if not isinstance(payload, dict):
            unreadable.append(path.relative_to(root_path).as_posix())
            continue
        payload["provenance_status"] = "stale"
        payload["stale_reason"] = normalized_reason
        stale_fields = set(str(value) for value in payload.get("stale_fields", ()))
        stale_fields.add("content_tree_sha256")
        payload["stale_fields"] = sorted(stale_fields)
        if replacement is not None:
            payload["replacement_artifact"] = replacement
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        updated.append(path.relative_to(root_path).as_posix())
    marker: dict[str, object] = {
        "schema_version": "stale-artifact-tree-v1",
        "provenance_status": "stale",
        "reason": normalized_reason,
        "replacement": replacement,
        "pre_invalidation_manifest_set_sha256": content_sha256(before),
        "manifests_updated": updated,
        "unreadable_manifests": unreadable,
    }
    (root_path / "STALE.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return marker
