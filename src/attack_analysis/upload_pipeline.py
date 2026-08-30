from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Callable, Iterable

from .browser_video import ensure_browser_compatible_video
from .pipeline import run_video_pipeline, validate_video_pipeline_source_contract
from .geometry import Projector
from .provenance import (
    MANIFEST_VERSION,
    SourceTier,
    build_provenance,
    compare_manifest_provenance,
    content_sha256,
    file_sha256,
    mark_legacy_manifest,
    validate_source_tier,
)
from .vision_pipeline import VisionConfig


VISION_CACHE_SCHEMA = "upload-vision-v3"


def _model_fingerprints(
    config: VisionConfig,
    xg_model_path: Path | None,
) -> dict[str, str]:
    references = {
        "pose": config.pose_model,
        "detection": config.detection_model,
        "ball_detection": config.ball_detection_model,
        "xg": str(xg_model_path) if xg_model_path is not None else None,
    }
    result: dict[str, str] = {}
    for name, reference in references.items():
        if reference is None:
            result[name] = content_sha256(None)
            continue
        path = Path(reference)
        result[name] = file_sha256(path) if path.is_file() else content_sha256(reference)
    return result


def _output_fingerprints(manifest: dict[str, Any]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name, value in sorted(dict(manifest.get("outputs") or {}).items()):
        if name == "manifest" or not value:
            continue
        path = Path(str(value))
        if path.is_file():
            fingerprints[str(name)] = file_sha256(path)
    return fingerprints


def _outputs_are_current(manifest: dict[str, Any]) -> bool:
    expected = dict(manifest.get("output_sha256") or {})
    if not expected:
        return False
    return expected == _output_fingerprints(manifest)


def _accepts_runner_keyword(runner: Callable[..., object], name: str) -> bool:
    """Preserve small test runners while forwarding the full contract to real runners."""

    try:
        parameters = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return runner is run_video_pipeline
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def vision_cache_key(
    video_digest: str,
    config: VisionConfig,
    cache_context: dict[str, object] | None = None,
) -> str:
    payload = {
        "schema": VISION_CACHE_SCHEMA,
        "video_digest": video_digest,
        "config": config,
        "context": cache_context or {},
    }
    return content_sha256(payload)


def run_cached_video_pipeline(
    input_path: Path,
    cache_root: Path,
    video_digest: str,
    config: VisionConfig,
    *,
    projector: Projector | None = None,
    xg_model_path: Path | None = None,
    cache_context: dict[str, object] | None = None,
    source_tier: SourceTier = "video_only",
    dataset_paths: Iterable[Path] = (),
    schema_versions: dict[str, object] | None = None,
    postprocess_config: dict[str, object] | None = None,
    content_root: Path | None = None,
    runner: Callable[..., dict[str, object]] = run_video_pipeline,
    browser_encoder: Callable[[Path], Path] = ensure_browser_compatible_video,
) -> dict[str, Any]:
    """Run YOLO once per exact video/model/config combination and reuse complete artifacts."""
    source_tier = validate_source_tier(str(source_tier))
    resolved_dataset_paths = tuple(Path(path) for path in dataset_paths)
    validate_video_pipeline_source_contract(
        source_tier=source_tier,
        projector=projector,
        dataset_paths=resolved_dataset_paths,
    )
    actual_video_digest = file_sha256(input_path)
    model_fingerprints = _model_fingerprints(config, xg_model_path)
    context = cache_context or {}
    provenance = build_provenance(
        source_tier=source_tier,
        video_sha256=actual_video_digest,
        config=config,
        model_fingerprints=model_fingerprints,
        schema_versions={
            "vision_cache": VISION_CACHE_SCHEMA,
            **(schema_versions or {}),
        },
        postprocess={"cache_context": context, **(postprocess_config or {})},
        dataset_paths=resolved_dataset_paths,
        content_root=content_root or Path(__file__).resolve().parent,
    )
    key = content_sha256(
        {
            "cache_schema": VISION_CACHE_SCHEMA,
            "artifact_identity": provenance["artifact_identity_sha256"],
        }
    )
    run_directory = cache_root / actual_video_digest / key[:16]
    manifest_path = run_directory / "manifest.json"
    if manifest_path.is_file():
        manifest = dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        comparison = compare_manifest_provenance(
            manifest,
            expected_provenance=provenance,
            source_tier=source_tier,
        )
        annotated = Path(str((manifest.get("outputs") or {}).get("annotated_video") or ""))
        outputs_current = _outputs_are_current(manifest)
        if (
            comparison.is_current
            and manifest.get("cache_key") == key
            and annotated.is_file()
            and outputs_current
        ):
            browser_video = browser_encoder(annotated)
            if browser_video != annotated:
                manifest["outputs"]["annotated_video_raw"] = str(annotated.resolve())
                manifest["outputs"]["annotated_video"] = str(browser_video.resolve())
                manifest["output_sha256"] = _output_fingerprints(manifest)
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            manifest["provenance_status"] = "validated"
            manifest["cache_hit"] = True
            return manifest
        # Legacy artifacts are explicitly classified, then rejected for cache reuse.
        manifest = mark_legacy_manifest(manifest)
        stale_fields = list(comparison.mismatches)
        if manifest.get("cache_key") != key:
            stale_fields.append("cache_key")
        if not annotated.is_file():
            stale_fields.append("outputs.annotated_video")
        if not outputs_current:
            stale_fields.append("output_sha256")
        if comparison.status == "legacy_unvalidated":
            manifest["provenance_status"] = "legacy_unvalidated"
        else:
            manifest["provenance_status"] = "stale"
        manifest["stale_fields"] = sorted(set(stale_fields))
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    runner_options: dict[str, object] = {}
    if projector is not None:
        runner_options["projector"] = projector
    if xg_model_path is not None:
        runner_options["xg_model_path"] = xg_model_path
    if _accepts_runner_keyword(runner, "source_tier"):
        runner_options["source_tier"] = source_tier
    if _accepts_runner_keyword(runner, "dataset_paths"):
        runner_options["dataset_paths"] = resolved_dataset_paths
    manifest = dict(runner(input_path, run_directory, config, **runner_options))
    manifest.setdefault("run_directory", str(run_directory.resolve()))
    annotated = Path(str((manifest.get("outputs") or {}).get("annotated_video") or ""))
    if not annotated.is_file():
        raise ValueError("video pipeline did not produce its declared annotated video")
    browser_video = browser_encoder(annotated)
    if browser_video != annotated:
        manifest["outputs"]["annotated_video_raw"] = str(annotated.resolve())
        manifest["outputs"]["annotated_video"] = str(browser_video.resolve())
    outputs = dict(manifest.get("outputs") or {})
    outputs.setdefault("manifest", str(manifest_path.resolve()))
    manifest["outputs"] = outputs
    manifest["manifest_version"] = MANIFEST_VERSION
    manifest["source_tier"] = source_tier
    manifest["provenance"] = provenance
    manifest["provenance_status"] = "validated"
    manifest["cache_schema"] = VISION_CACHE_SCHEMA
    manifest["cache_key"] = key
    manifest["video_digest"] = actual_video_digest
    manifest["caller_video_digest"] = video_digest
    manifest["cache_hit"] = False
    manifest["output_sha256"] = _output_fingerprints(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
