import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.prediction_sources import (
    OfficialScore,
    PredictionSource,
    PredictionSourceError,
    load_registry,
    parse_official_output,
    register_prediction_source,
    resolve_prediction_source,
)

SAMPLE_OUTPUT = """
=== Aggregated Precision/Recall per Class (ALL GAMES) ===
Class      Precision     Recall      TP      FP      FN      GT
-------------------------------------------------------------
drive          0.700      0.650     100      40      54     154
-------------------------------------------------------------
OVERALL        0.733      0.706    4300    1500    1790    6090       1       2       3       4       5       6       7       8       9      10      11      12
Micro F1: 0.719
"""


def score(f1: float) -> OfficialScore:
    return OfficialScore(
        micro_f1=f1,
        precision=f1,
        recall=f1,
        true_positives=1,
        false_positives=1,
        false_negatives=1,
        ground_truth=2,
    )


class PredictionSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.registry = self.root / "models" / "prediction-sources.json"
        self.store = self.root / "FOOTPASS" / "playbyplay_PRED"
        self.store.mkdir(parents=True)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _json(self, name: str, payload: object, directory: Path | None = None) -> Path:
        path = (directory or self.store) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _register(self, path: Path, source_id: str, kind: str, f1: float) -> dict:
        return register_prediction_source(
            path,
            source_id=source_id,
            kind=kind,
            score=score(f1),
            registry_path=self.registry,
            store_directory=self.store,
        )

    def test_parses_official_evaluator_output(self) -> None:
        parsed = parse_official_output(SAMPLE_OUTPUT)
        self.assertEqual(parsed.micro_f1, 0.719)
        self.assertEqual(parsed.precision, 0.733)
        self.assertEqual(parsed.true_positives, 4300)
        self.assertEqual(parsed.ground_truth, 6090)

    def test_missing_metric_lines_are_rejected(self) -> None:
        with self.assertRaises(PredictionSourceError):
            parse_official_output("nothing useful")

    def test_first_dst_becomes_primary_and_resolves_by_default(self) -> None:
        result = self._register(self._json("playbyplay_DST_val.json", {"a": 1}), "DST_v1", "DST", 0.719)
        self.assertEqual(result["role"], "primary")
        source, path = resolve_prediction_source(None, self.registry)
        self.assertEqual(source.id, "DST_v1")
        self.assertEqual(path, (self.store / "playbyplay_DST_val.json").resolve())

    def test_raw_taad_is_always_ablation(self) -> None:
        result = self._register(self._json("taad.json", {"a": 2}), "TAAD_h100", "TAAD", 0.99)
        self.assertEqual(result["role"], "ablation")
        with self.assertRaises(PredictionSourceError):
            PredictionSource(id="x", kind="TAAD", path="p", sha256="0" * 64, role="primary")

    def test_dst_must_strictly_beat_primary(self) -> None:
        self._register(self._json("playbyplay_DST_val.json", {"a": 1}), "DST_v1", "DST", 0.719)
        outside = self.root / "downloads"
        tie = self._register(self._json("dst_tie.json", {"a": 3}, outside), "DST_tie", "DST", 0.719)
        self.assertEqual(tie["role"], "ablation")
        better = self._register(self._json("dst_new.json", {"a": 4}, outside), "DST_h100", "DST", 0.74)
        self.assertTrue(better["promoted"])
        self.assertEqual(better["previous_primary"], "DST_v1")
        self.assertTrue(better["warnings"])
        registry = load_registry(self.registry)
        self.assertEqual(registry.primary.id, "DST_h100")
        self.assertEqual(registry.get("DST_v1").role, "ablation")
        self.assertTrue((self.store / "playbyplay_DST_h100_val.json").is_file())
        self.assertEqual(json.loads((self.store / "playbyplay_DST_val.json").read_text()), {"a": 1})

    def test_changed_file_fails_hash_check(self) -> None:
        path = self._json("playbyplay_DST_val.json", {"a": 1})
        self._register(path, "DST_v1", "DST", 0.719)
        path.write_text(json.dumps({"a": "tampered"}), encoding="utf-8")
        with self.assertRaises(PredictionSourceError):
            resolve_prediction_source("DST_v1", self.registry)

    def test_never_overwrites_a_different_stored_file(self) -> None:
        self._json("playbyplay_DST_h100_val.json", {"old": True})
        incoming = self._json("dst.json", {"new": True}, self.root / "downloads")
        with self.assertRaises(PredictionSourceError):
            self._register(incoming, "DST_h100", "DST", 0.8)

    def test_duplicate_id_is_rejected(self) -> None:
        path = self._json("playbyplay_DST_val.json", {"a": 1})
        self._register(path, "DST_v1", "DST", 0.719)
        with self.assertRaises(PredictionSourceError):
            self._register(path, "DST_v1", "DST", 0.8)


if __name__ == "__main__":
    unittest.main()
