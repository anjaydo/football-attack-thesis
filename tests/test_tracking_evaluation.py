import math
import unittest

from attack_analysis.tracking_evaluation import (
    TrackDetection,
    TrackingBox,
    evaluate_tracking,
    optimal_iou_matches,
)


def _box(x1: float = 0.0, x2: float = 10.0) -> TrackingBox:
    return TrackingBox(x1, 0.0, x2, 10.0)


def _player(
    frame: int,
    track: str | None,
    *,
    team: str | None = "home",
    jersey: int | None = 9,
    bbox: TrackingBox | None = None,
    confidence: float = 1.0,
) -> TrackDetection:
    return TrackDetection(
        match_id="game_6",
        period=1,
        frame_index=frame,
        bbox=bbox or _box(),
        object_type="player",
        track_id=track,
        team_id=team,
        jersey_number=jersey,
        confidence=confidence,
    )


def _ball(frame: int, track: str | None, *, bbox: TrackingBox | None = None) -> TrackDetection:
    return TrackDetection(
        match_id="game_6",
        period=1,
        frame_index=frame,
        bbox=bbox or TrackingBox(20.0, 20.0, 22.0, 22.0),
        object_type="ball",
        track_id=track,
    )


class TrackingEvaluationTests(unittest.TestCase):
    def test_perfect_tracks_score_one(self) -> None:
        ground_truth = [
            _player(0, "gt-9"),
            _player(1, "gt-9"),
            _player(0, "gt-4", team="away", jersey=4, bbox=_box(30, 40)),
            _player(1, "gt-4", team="away", jersey=4, bbox=_box(30, 40)),
            _ball(0, "gt-ball"),
            _ball(1, "gt-ball"),
        ]
        predictions = [
            _player(0, "pred-9"),
            _player(1, "pred-9"),
            _player(0, "pred-4", team="away", jersey=4, bbox=_box(30, 40)),
            _player(1, "pred-4", team="away", jersey=4, bbox=_box(30, 40)),
            _ball(0, "pred-ball"),
            _ball(1, "pred-ball"),
        ]

        report = evaluate_tracking(ground_truth, predictions)

        self.assertEqual(report.player_detection.f1, 1.0)
        self.assertEqual(report.player_detection.ap50, 1.0)
        self.assertEqual(report.player_identity.idf1, 1.0)
        self.assertEqual(report.player_hota.hota, 1.0)
        self.assertEqual(report.team.accuracy, 1.0)
        self.assertEqual(report.jersey.accuracy, 1.0)
        self.assertEqual(report.ball_track_recall, 1.0)
        self.assertEqual(report.bbox_gs_hota_style.hota, 1.0)

    def test_id_switch_reduces_idf1_and_association_not_detection(self) -> None:
        ground_truth = [_player(frame, "gt-9") for frame in range(4)]
        predictions = [
            _player(frame, "pred-a" if frame < 2 else "pred-b")
            for frame in range(4)
        ]

        report = evaluate_tracking(ground_truth, predictions)

        self.assertEqual(report.player_detection.f1, 1.0)
        self.assertEqual(report.player_identity.idf1, 0.5)
        self.assertAlmostEqual(report.player_hota.assa or -1.0, 0.5)
        self.assertAlmostEqual(report.player_hota.hota or -1.0, math.sqrt(0.5))

    def test_team_error_is_visible_and_gates_bbox_gs_hota_style(self) -> None:
        ground_truth = [_player(frame, "gt-9") for frame in range(2)]
        predictions = [
            _player(frame, "pred-9", team="away") for frame in range(2)
        ]

        report = evaluate_tracking(ground_truth, predictions)

        self.assertEqual(report.player_hota.hota, 1.0)
        self.assertEqual(report.team.accuracy, 0.0)
        self.assertEqual(report.team.prediction_coverage, 1.0)
        self.assertEqual(report.bbox_gs_hota_style.hota, 0.0)

    def test_missing_ball_is_not_hidden_by_player_performance(self) -> None:
        ground_truth = [_player(0, "gt-9"), _ball(0, "gt-ball"), _ball(1, "gt-ball")]
        predictions = [_player(0, "pred-9")]

        report = evaluate_tracking(ground_truth, predictions)

        self.assertEqual(report.player_detection.recall, 1.0)
        self.assertEqual(report.ball_detection.recall, 0.0)
        self.assertEqual(report.ball_detection.ap50, 0.0)
        self.assertEqual(report.ball_track_recall, 0.0)
        self.assertEqual(report.visible_ball_ground_truth, 2)

    def test_empty_input_reports_unavailable_metrics_not_fake_zeroes(self) -> None:
        report = evaluate_tracking([], [])

        self.assertIsNone(report.player_detection.precision)
        self.assertIsNone(report.player_detection.recall)
        self.assertIsNone(report.player_detection.f1)
        self.assertIsNone(report.player_detection.ap50)
        self.assertIsNone(report.player_identity.idf1)
        self.assertIsNone(report.player_hota.hota)
        self.assertIsNone(report.team.accuracy)
        self.assertIsNone(report.ball_track_recall)
        self.assertIsNone(report.bbox_gs_hota_style.hota)

    def test_unknown_gold_attributes_are_excluded_not_encoded_as_zero(self) -> None:
        ground_truth = [_player(0, "gt-9", team=None, jersey=None)]
        predictions = [_player(0, "pred-9", team="away", jersey=88)]

        report = evaluate_tracking(ground_truth, predictions)

        self.assertEqual(report.player_hota.hota, 1.0)
        self.assertIsNone(report.team.accuracy)
        self.assertIsNone(report.jersey.accuracy)
        self.assertEqual(report.gs_team_label_coverage, 0.0)
        self.assertEqual(report.gs_jersey_label_coverage, 0.0)
        self.assertEqual(report.bbox_gs_hota_style.hota, 1.0)

    def test_missing_predicted_known_attribute_is_an_error_and_abstention(self) -> None:
        report = evaluate_tracking(
            [_player(0, "gt-9", team="home")],
            [_player(0, "pred-9", team=None)],
        )

        self.assertEqual(report.team.accuracy, 0.0)
        self.assertEqual(report.team.prediction_coverage, 0.0)
        self.assertEqual(report.bbox_gs_hota_style.hota, 0.0)

    def test_optimal_matching_maximizes_cardinality_before_iou(self) -> None:
        # Greedy matching would consume prediction 0 for target 0 (IoU .818)
        # and leave target 1 unmatched.  The optimum makes both valid matches.
        ground_truth = [
            _player(0, "gt-a", bbox=_box(0, 10)),
            _player(0, "gt-b", bbox=_box(4, 14)),
        ]
        predictions = [
            _player(0, "pred-a", bbox=_box(1, 11)),
            _player(0, "pred-b", bbox=_box(-2, 8)),
        ]

        matches = optimal_iou_matches(ground_truth, predictions, 0.5)

        self.assertEqual(len(matches), 2)
        self.assertEqual(
            {(match.ground_truth_index, match.prediction_index) for match in matches},
            {(0, 1), (1, 0)},
        )

    def test_explicitly_invisible_ground_truth_is_outside_evaluation(self) -> None:
        invisible = TrackDetection(
            match_id="game_6",
            frame_index=0,
            bbox=_box(),
            visibility=False,
            track_id="gt",
        )

        report = evaluate_tracking([invisible], [])

        self.assertEqual(report.player_detection.ground_truth, 0)
        self.assertIsNone(report.player_detection.recall)

    def test_rejects_invalid_data_and_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            TrackingBox(0, 0, 0, 1)
        with self.assertRaises(ValueError):
            TrackDetection("game", 0, _box(), confidence=1.1)
        with self.assertRaises(ValueError):
            evaluate_tracking([], [], primary_iou_threshold=0.0)
        with self.assertRaises(ValueError):
            evaluate_tracking([], [], hota_iou_thresholds=())


if __name__ == "__main__":
    unittest.main()
