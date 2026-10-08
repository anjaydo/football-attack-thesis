import unittest

from attack_analysis.environment import (
    default_inference_imgsz,
    default_torch_device,
    environment_report,
)


class EnvironmentReportTests(unittest.TestCase):
    def test_report_contains_readiness_and_recommendations(self) -> None:
        report = environment_report()
        self.assertIn("python", report)
        self.assertIn("packages", report)
        self.assertIn("vision_ready", report)
        self.assertIsInstance(report["recommendations"], list)

    def test_cpu_vision_recommends_gpu_for_ball_finetune(self) -> None:
        report = environment_report()
        if report["packages"]["torch"] is None or not report["vision_ready"]:
            self.skipTest("vision extras not installed")
        if report["tools"]["cuda_available"]:
            self.skipTest("CUDA available on this machine")
        joined = " ".join(report["recommendations"])
        self.assertIn("imgsz 1280", joined)

    def test_cuda_inference_uses_1280(self) -> None:
        if default_torch_device() != "0":
            self.assertEqual(default_inference_imgsz(), 960)
            return
        self.assertEqual(default_inference_imgsz(), 1280)


if __name__ == "__main__":
    unittest.main()

