import unittest

from pluribus_ri.abstraction import (
    build_postflop_bucket_calibration_report,
    compare_postflop_bucket_policies,
)


class AbstractionMetricsTests(unittest.TestCase):
    def test_postflop_bucket_calibration_report_is_deterministic(self) -> None:
        report_a = build_postflop_bucket_calibration_report(
            policy="texture_v1",
            samples=300,
            seed=123,
            board_lengths=(3, 4, 5),
            preflop_bucket_policy="canonical169",
        )
        report_b = build_postflop_bucket_calibration_report(
            policy="texture_v1",
            samples=300,
            seed=123,
            board_lengths=(3, 4, 5),
            preflop_bucket_policy="canonical169",
        )

        self.assertEqual(report_a, report_b)
        self.assertEqual(report_a.samples, 300)
        self.assertGreater(report_a.unique_buckets, 0)
        self.assertGreaterEqual(report_a.bucket_entropy, 0.0)
        self.assertGreaterEqual(report_a.max_bucket_size, 1)
        self.assertGreaterEqual(report_a.mean_bucket_score_std, 0.0)
        self.assertGreaterEqual(report_a.weighted_bucket_score_std, 0.0)
        self.assertGreaterEqual(report_a.score_bucket_spearman, -1.0)
        self.assertLessEqual(report_a.score_bucket_spearman, 1.0)

    def test_compare_postflop_bucket_policies(self) -> None:
        reports = compare_postflop_bucket_policies(
            policies=("legacy", "texture_v1"),
            samples=200,
            seed=7,
        )
        self.assertEqual(set(reports.keys()), {"legacy", "texture_v1"})
        self.assertNotEqual(
            reports["legacy"].unique_buckets,
            reports["texture_v1"].unique_buckets,
        )


if __name__ == "__main__":
    unittest.main()
