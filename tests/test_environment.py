import unittest

from attack_analysis.environment import environment_report


class EnvironmentReportTests(unittest.TestCase):
    def test_report_contains_readiness_and_recommendations(self) -> None:
        report = environment_report()
        self.assertIn("python", report)
        self.assertIn("packages", report)
        self.assertIn("vision_ready", report)
        self.assertIsInstance(report["recommendations"], list)

