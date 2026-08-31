from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from attack_analysis.attack_v2 import decode_footpass_attack_v2
from attack_analysis.benchmarks import FrozenBenchmarkStore, write_historical_snapshot
from attack_analysis.fusion import associate_action_with_frame, bbox_iou, score_attack_with_observations
from attack_analysis.models import ObservedActionV2, SourceTier, TerminalType
from attack_analysis.observations import BoundingBox, FrameObservation, TrackedObject
from attack_analysis.overview import attack_v2_research_overview
from attack_analysis.research import BenchmarkRecord


SOURCE = SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL


def _action(second: float, action_type: str, team: str, *, x: float = 90.0, y: float = 34.0) -> ObservedActionV2:
    return ObservedActionV2(
        match_id="game_18",
        period=1,
        frame=round(second * 25),
        timestamp_sec=second,
        action_type=action_type,
        source_tier=SOURCE,
        actor_team_id=team,
        x=x,
        y=y,
        evidence_ids=(f"ev-{second}-{action_type}",),
    )


def _player(track_id: str, team: str, x: float, y: float, *, goalkeeper: bool = False) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        object_type="goalkeeper" if goalkeeper else "player",
        confidence=0.9,
        bbox=BoundingBox(10, 10, 30, 50),
        team=team,
        pitch_x=x,
        pitch_y=y,
    )


class FusionAndOverviewTests(unittest.TestCase):
    def test_associates_nearest_player_and_fills_xg_context(self) -> None:
        action = _action(1.0, "shot", "A")
        frame = FrameObservation(
            frame_index=25,
            timestamp_sec=1.0,
            objects=(
                _player("a1", "team_a", 90.2, 34.1),
                _player("b1", "team_b", 100.0, 34.0, goalkeeper=True),
                _player("b2", "team_b", 91.0, 34.0),
            ),
        )
        association = associate_action_with_frame(
            action,
            frame,
            team_mapping={"A": "team_a"},
        )
        self.assertEqual(association.track_id, "a1")
        self.assertEqual(association.method, "pitch_nearest")
        self.assertEqual(association.xg_raw["goalkeeper_visible"], True)
        self.assertEqual(association.xg_raw["foe_players"], 2)

    def test_infers_goalkeeper_body_part_and_play_pattern(self) -> None:
        throw_in = _action(1.0, "throw_in", "A", x=70.0)
        shot = _action(3.0, "shot", "A", x=90.0)
        keypoints = [(0.0, 0.0, 0.0)] * 17
        keypoints[0] = (12.0, 10.0, 0.9)
        keypoints[16] = (20.0, 48.0, 0.9)
        shooter = TrackedObject(
            track_id="a1",
            object_type="player",
            confidence=0.9,
            bbox=BoundingBox(10, 10, 40, 50),
            team="team_a",
            pitch_x=90.2,
            pitch_y=34.1,
            keypoints=tuple(tuple(value) for value in keypoints),
        )
        frame = FrameObservation(
            frame_index=75,
            timestamp_sec=3.0,
            objects=(
                shooter,
                TrackedObject(
                    track_id="b1",
                    object_type="player",
                    confidence=0.9,
                    bbox=BoundingBox(80, 10, 100, 50),
                    team="team_b",
                    pitch_x=102.0,
                    pitch_y=34.0,
                ),
                TrackedObject(
                    track_id="ball",
                    object_type="ball",
                    confidence=0.8,
                    bbox=BoundingBox(18, 44, 26, 52),
                ),
            ),
        )
        association = associate_action_with_frame(
            shot,
            frame,
            team_mapping={"A": "team_a"},
            possession_actions=(throw_in, shot),
        )
        self.assertEqual(association.xg_raw["body_part"], "right_foot")
        self.assertEqual(association.xg_raw["play_pattern"], "throw_in")
        self.assertEqual(association.xg_raw["shot_type"], "open_play")
        self.assertTrue(association.xg_raw["goalkeeper_visible"])
        self.assertAlmostEqual(association.xg_raw["x_goalkeeper"], 102.0)

    def test_bbox_iou_and_ball_gate_warns_but_keeps_location_xg(self) -> None:
        self.assertGreater(
            bbox_iou(BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15)),
            0.0,
        )
        attack = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        ).attacks[0]
        frame = FrameObservation(
            frame_index=50,
            timestamp_sec=2.0,
            objects=(_player("a1", "team_a", 90.0, 34.0),),
        )
        result = score_attack_with_observations(attack, frame, ball_f1=0.34)
        self.assertIsNone(result.xg_record)
        self.assertEqual(result.withheld_reason, "no_xg_v2_model")
        self.assertIn("ball F1", str(result.association.spatial_context.get("ball_quality_warning")))
        no_model = score_attack_with_observations(attack, frame)
        self.assertEqual(no_model.withheld_reason, "no_xg_v2_model")

    def test_possession_buildup_uses_latest_cross_as_assist(self) -> None:
        from attack_analysis.fusion import possession_buildup

        buildup = possession_buildup(
            (
                _action(1.0, "pass", "A", x=70.0),
                _action(2.0, "cross", "A", x=88.0, y=10.0),
                _action(3.0, "header", "A", x=94.0),
            )
        )
        self.assertEqual(buildup["assist_type"], "cross")
        self.assertTrue(buildup["cross_then_header"])
        self.assertEqual(len(buildup["events"]), 3)

    def test_infers_body_part_from_pose_without_ball(self) -> None:
        shot = _action(3.0, "shot", "A", x=90.0)
        keypoints = [(0.0, 0.0, 0.0)] * 17
        keypoints[16] = (20.0, 48.0, 0.9)
        shooter = TrackedObject(
            track_id="a1",
            object_type="player",
            confidence=0.9,
            bbox=BoundingBox(10, 10, 40, 50),
            team="team_a",
            pitch_x=90.2,
            pitch_y=34.1,
            keypoints=tuple(tuple(value) for value in keypoints),
        )
        frame = FrameObservation(frame_index=75, timestamp_sec=3.0, objects=(shooter,))
        association = associate_action_with_frame(shot, frame, team_mapping={"A": "team_a"})
        self.assertEqual(association.xg_raw["body_part"], "right_foot")

    def test_footpass_freeze_overrides_yolo_counts(self) -> None:
        attack = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        ).attacks[0]
        frame = FrameObservation(
            frame_index=50,
            timestamp_sec=2.0,
            objects=(_player("a1", "team_a", 90.0, 34.0),),
        )
        freeze = {
            "x_shooter": 91.0,
            "y_shooter": 34.0,
            "x_goalkeeper": 103.0,
            "y_goalkeeper": 34.0,
            "foe_players": 11,
            "ally_players": 10,
            "defending_players": 2,
            "under_pressure": 1,
            "observed_players": 22,
            "gk_distance_m": 12.0,
            "defenders_in_box": 4,
            "allies_ahead": 1,
            "foes_in_shot_corridor": 2,
            "players": [
                {"side": "foe", "role": "goalkeeper", "x": 103.0, "y": 34.0, "is_goalkeeper": True, "is_shooter": False}
            ],
            "evidence_id": "footpass-hdf5:game_18_H1:50:freeze",
        }
        gated = score_attack_with_observations(attack, frame, ball_f1=0.34, footpass_freeze=freeze)
        self.assertIsNone(gated.xg_record)
        self.assertEqual(gated.withheld_reason, "no_xg_v2_model")
        self.assertEqual(gated.association.xg_raw["foe_players"], 11)
        self.assertEqual(gated.association.xg_raw["x_goalkeeper"], 103.0)
        self.assertTrue(gated.association.xg_raw["goalkeeper_visible"])
        self.assertEqual(gated.association.spatial_context["gk_distance_m"], 12.0)
        self.assertEqual(
            gated.association.spatial_context["possession_buildup"]["assist_type"],
            "pass",
        )
        self.assertEqual(
            gated.association.spatial_context["provenance"]["foe_players"],
            "footpass_tactical",
        )

    def test_votes_yolo_team_from_nearby_tracks(self) -> None:
        from attack_analysis.fusion import vote_team_mapping

        action = _action(1.0, "shot", "A")
        frame = FrameObservation(
            frame_index=25,
            timestamp_sec=1.0,
            objects=(
                _player("b1", "team_b", 90.1, 34.0),
                _player("a1", "team_a", 40.0, 20.0),
            ),
        )
        mapping = vote_team_mapping(action, frame, opposing_team_id="B")
        self.assertEqual(mapping["A"], "team_b")
        self.assertEqual(mapping["B"], "team_a")

    def test_play_pattern_heuristics_corner_free_kick_and_penalty(self) -> None:
        from attack_analysis.fusion import play_pattern_from_possession, shot_type_from_possession

        corner = (
            _action(1.0, "cross", "A", x=104.0, y=2.0),
            _action(2.0, "header", "A", x=94.0, y=34.0),
        )
        self.assertEqual(play_pattern_from_possession(corner), "corner")
        self.assertEqual(shot_type_from_possession(corner), "open_play")
        free_kick = (
            _action(1.0, "shot", "A", x=78.0, y=40.0),
        )
        self.assertEqual(play_pattern_from_possession(free_kick), "free_kick")
        self.assertEqual(shot_type_from_possession(free_kick), "free_kick")
        penalty = (_action(1.0, "shot", "A", x=99.5, y=34.0),)
        self.assertEqual(shot_type_from_possession(penalty), "penalty")

    def test_orients_yolo_when_left_to_right_flag_is_missing(self) -> None:
        action = _action(1.0, "shot", "A", x=90.0, y=34.0)
        self.assertIsNone(action.raw_left_to_right)
        frame = FrameObservation(
            frame_index=25,
            timestamp_sec=1.0,
            objects=(
                _player("a1", "team_a", 15.0, 34.0),
                _player("b1", "team_b", 5.0, 34.0, goalkeeper=True),
            ),
        )
        association = associate_action_with_frame(action, frame, team_mapping={"A": "team_a"})
        self.assertEqual(association.track_id, "a1")
        self.assertEqual(association.method, "pitch_nearest")
        self.assertAlmostEqual(association.xg_raw["x_goalkeeper"], 100.0)

    def test_aligns_yolo_tracks_to_freeze_roles(self) -> None:
        from attack_analysis.fusion import align_yolo_tracks_to_freeze

        aligned = align_yolo_tracks_to_freeze(
            (
                _player("a1", "team_a", 90.0, 34.0),
                _player("b1", "team_b", 102.0, 34.0),
            ),
            (
                {"side": "ally", "role": "central_forward", "x": 90.2, "y": 34.1, "is_goalkeeper": False},
                {"side": "foe", "role": "goalkeeper", "x": 102.1, "y": 34.0, "is_goalkeeper": True},
            ),
        )
        self.assertEqual(aligned["n_matched"], 2)
        self.assertEqual(aligned["agreement"], 1.0)
        self.assertTrue(aligned["goalkeeper_in_camera"])
        by_track = {item["track_id"]: item for item in aligned["assignments"]}
        self.assertEqual(by_track["a1"]["freeze_side"], "ally")
        self.assertEqual(by_track["b1"]["freeze_role"], "goalkeeper")

    def test_projects_pose_onto_pitch_from_bbox_anchor(self) -> None:
        from attack_analysis.fusion import project_pose_to_pitch

        keypoints = [(0.0, 0.0, 0.0)] * 17
        keypoints[16] = (20.0, 50.0, 0.9)
        shooter = TrackedObject(
            track_id="a1",
            object_type="player",
            confidence=0.9,
            bbox=BoundingBox(10, 10, 30, 50),
            team="team_a",
            pitch_x=90.0,
            pitch_y=34.0,
            keypoints=tuple(tuple(value) for value in keypoints),
        )
        pose = project_pose_to_pitch(shooter)
        ankle = next(item for item in pose if item["index"] == 16)
        self.assertEqual(ankle["pitch_source"], "local_affine")
        self.assertAlmostEqual(float(ankle["pitch_x"]), 90.0, places=1)
        self.assertAlmostEqual(float(ankle["pitch_y"]), 34.0, places=1)

    def test_research_overview_abstains_from_effectiveness(self) -> None:
        attack = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        ).attacks[0]
        overview = attack_v2_research_overview(attack)
        self.assertTrue(overview["abstained_effectiveness"])
        self.assertIsNone(overview["effectiveness_verdict"])
        self.assertEqual(overview["xg_status"], "unknown")
        self.assertEqual(overview["terminal_type"], TerminalType.SHOT_ATTEMPT.value)

    def test_historical_snapshot_is_frozen_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.jsonl"
            records = write_historical_snapshot(
                [
                    {
                        "role": "central_forward",
                        "metric": "xg",
                        "value_per90": 0.35,
                        "percentile": 80,
                        "sample_size": 20,
                    }
                ],
                path,
                snapshot_date="2026-08-30",
                default_season="2023-24",
                default_source_url="https://example.test/stats",
            )
            store = FrozenBenchmarkStore.from_jsonl(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(store.search("xg")[0].source_url, "https://example.test/stats")
        self.assertIsInstance(store.records[0], BenchmarkRecord)
