from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

from .attack_v2 import decode_footpass_attack_v2
from .footpass import footpass_match_keys, prepare_footpass_actions_v2
from .models import ATTACK_QUALIFYING_STATUSES, ATTACK_TERMINAL_ALLOWLIST
from .provenance import (
    MANIFEST_VERSION,
    build_provenance,
    content_sha256,
    files_sha256,
)


def audit_footpass_v2(
    playbyplay_path: Path,
    tactical_h5: Path,
    video_root: Path,
    output_path: Path,
) -> dict[str, object]:
    """Regenerate the oracle-assisted validation audit from current geometry."""

    keys = footpass_match_keys(playbyplay_path)
    game_ids = sorted({key.split("_H", 1)[0] for key in keys})
    videos = tuple(video_root / f"{game_id}.mp4" for game_id in game_ids)
    missing_videos = [str(path) for path in videos if not path.is_file()]
    if missing_videos:
        raise FileNotFoundError("missing validation videos: " + ", ".join(missing_videos))

    halves: list[dict[str, object]] = []
    shot_x: list[float] = []
    regression_frame_x: list[float] = []
    all_attacks = []
    total_actions = total_possessions = total_replays = total_unassigned = 0
    headed_promoted = blocks_promoted = headers_left = blocks_left = 0
    for key in keys:
        prepared = prepare_footpass_actions_v2(
            playbyplay_path,
            match_key=key,
            source="oracle",
            tactical_h5=tactical_h5,
        )
        actions = prepared.actions
        decoded = decode_footpass_attack_v2(actions)
        shots = [item for item in actions if item.action_type == "shot" and item.x is not None]
        shot_x.extend(float(item.x) for item in shots)
        if key == "game_18_H1":
            regression_frame_x.extend(
                float(item.x)
                for item in actions
                if item.frame == 2640 and item.action_type == "shot" and item.x is not None
            )
        all_attacks.extend(decoded.attacks)
        total_actions += len(actions)
        total_possessions += len(decoded.possessions)
        total_replays += len(decoded.excluded_replay_action_ids)
        total_unassigned += len(decoded.unassigned_action_ids)
        headed_promoted += prepared.headed_promoted
        blocks_promoted += prepared.blocks_promoted
        headers_left += prepared.headers_left_generic
        blocks_left += prepared.blocks_left_generic
        halves.append(
            {
                "match_key": key,
                "observed_actions": len(actions),
                "possessions": len(decoded.possessions),
                "strict_attacks": len(decoded.attacks),
                "excluded_replay_actions": len(decoded.excluded_replay_action_ids),
                "unassigned_actions": len(decoded.unassigned_action_ids),
                "headed_promoted": prepared.headed_promoted,
                "blocks_promoted": prepared.blocks_promoted,
                "headers_left_generic": prepared.headers_left_generic,
                "blocks_left_generic": prepared.blocks_left_generic,
                "attack_terminals": dict(
                    Counter(item.terminal.terminal_type.value for item in decoded.attacks)
                ),
            }
        )

    if not shot_x or len(regression_frame_x) != 1:
        raise ValueError("geometry regression evidence is incomplete")
    geometry = {
        "game_18_H1_frame_2640_shot_x": regression_frame_x[0],
        "oracle_shots_with_coordinates": len(shot_x),
        "oracle_shot_median_x": median(shot_x),
        "oracle_shot_fraction_x_ge_70": sum(value >= 70.0 for value in shot_x) / len(shot_x),
        "frame_2640_within_tolerance": abs(regression_frame_x[0] - 82.094) <= 0.01,
        "median_x_gate": median(shot_x) > 85.0,
        "x_ge_70_gate": sum(value >= 70.0 for value in shot_x) / len(shot_x) >= 0.95,
    }
    invariants = {
        "all_attacks_end_observed": all(
            item.possession.status in ATTACK_QUALIFYING_STATUSES for item in all_attacks
        ),
        "all_attacks_have_allowlisted_terminal": all(
            item.terminal.terminal_type in ATTACK_TERMINAL_ALLOWLIST for item in all_attacks
        ),
        "all_attack_ids_unique": len({item.attack_id for item in all_attacks}) == len(all_attacks),
        "no_effectiveness_verdicts_emitted": True,
        "geometry_regression_passed": all(
            geometry[key]
            for key in ("frame_2640_within_tolerance", "median_x_gate", "x_ge_70_gate")
        ),
    }
    source_tier = "oracle_actions_oracle_tactical"
    provenance = build_provenance(
        source_tier=source_tier,
        video_sha256=files_sha256(videos),
        config={"decoder": "deterministic-terminal-first-v2", "fps": 25.0},
        model_fingerprints={"structured_decoder": content_sha256("untrained-rule-baseline-v2")},
        schema_versions={"domain": "2.0", "audit": "footpass-oracle-audit-v2"},
        postprocess={"terminal_allowlist": sorted(item.value for item in ATTACK_TERMINAL_ALLOWLIST)},
        dataset_paths=(playbyplay_path, tactical_h5),
        content_root=Path(__file__).resolve().parent,
    )
    report: dict[str, object] = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": "footpass-oracle-audit-v2",
        "source_tier": source_tier,
        "release_status": "research_only",
        "provenance_status": "validated",
        "provenance": provenance,
        "summary": {
            "match_halves": len(keys),
            "observed_actions": total_actions,
            "possessions": total_possessions,
            "strict_attacks": len(all_attacks),
            "attack_terminals": dict(
                Counter(item.terminal.terminal_type.value for item in all_attacks)
            ),
            "headed_promoted": headed_promoted,
            "blocks_promoted": blocks_promoted,
            "headers_left_generic": headers_left,
            "blocks_left_generic": blocks_left,
            "excluded_replay_actions": total_replays,
            "unassigned_actions": total_unassigned,
        },
        "geometry": geometry,
        "invariants": invariants,
        "halves": halves,
        "limitations": [
            "Oracle action/tactical audit is assisted, not video-only.",
            "Raw FOOTPASS header/block remain generic unless neighborhood inference supplies headed_attempt or forced_goal_line_out evidence.",
            "No human attack-boundary gold is available yet; no accuracy claim is made.",
        ],
    }
    if not all(invariants.values()):
        raise ValueError("FOOTPASS v2 audit invariant failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


__all__ = ["audit_footpass_v2"]
