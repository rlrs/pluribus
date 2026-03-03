import unittest

from pluribus_ri.runtime_search import (
    LeafContinuationConfig,
    NestedUnsafeSearchConfig,
    RuntimeSearchBenchmarkConfig,
    SubgameStoppingRules,
    run_nested_search_benchmark,
)


class RuntimeSearchBenchmarkTests(unittest.TestCase):
    def test_benchmark_runs_and_reports_consistent_counts(self) -> None:
        result = run_nested_search_benchmark(
            RuntimeSearchBenchmarkConfig(
                runs=4,
                seed_start=40,
                max_prefix_actions=6,
                random_action_seed=99,
                max_sampling_attempts=200,
                search_config=NestedUnsafeSearchConfig(
                    random_seed=5,
                    stopping_rules=SubgameStoppingRules(
                        min_cfr_iterations=1,
                        max_cfr_iterations=1,
                        max_nodes_touched=20_000,
                        max_wallclock_ms=1000,
                        leaf_max_depth=6,
                    ),
                    leaf_continuation=LeafContinuationConfig(
                        rollout_count=1,
                        max_actions_per_rollout=64,
                        random_seed=17,
                    ),
                ),
            )
        )

        self.assertEqual(result.run_count, 4)
        self.assertGreaterEqual(result.attempted_states, result.run_count)
        self.assertEqual(len(result.seeds_used), 4)

        self.assertEqual(sum(result.stop_reason_counts.values()), 4)
        self.assertEqual(sum(result.street_counts.values()), 4)
        self.assertEqual(sum(result.chosen_action_kind_counts.values()), 4)

        self.assertGreaterEqual(result.latency_ms_mean, result.latency_ms_min)
        self.assertGreaterEqual(result.latency_ms_p50, result.latency_ms_min)
        self.assertGreaterEqual(result.latency_ms_p95, result.latency_ms_p50)
        self.assertGreaterEqual(result.latency_ms_p99, result.latency_ms_p95)
        self.assertGreaterEqual(result.latency_ms_max, result.latency_ms_p99)

        payload = result.to_dict()
        self.assertEqual(payload["run_count"], 4)
        self.assertIn("latency_ms", payload)
        self.assertIn("nodes_visited", payload)
        self.assertIn("cfr_iterations", payload)

    def test_benchmark_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            run_nested_search_benchmark(RuntimeSearchBenchmarkConfig(runs=0))


if __name__ == "__main__":
    unittest.main()
