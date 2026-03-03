import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pluribus_ri.blueprint import (
    AIVATConfig,
    BlueprintPolicy,
    ExploitabilityProxyConfig,
    LeagueEvaluationConfig,
    SUPPORTED_BASELINES,
    load_blueprint_policy,
    run_exploitability_proxy_report,
    run_one_vs_field_league,
)


class BlueprintEvaluationTests(unittest.TestCase):
    def test_one_vs_field_league_reports_matchups_matrix_and_aggregates(self) -> None:
        policy_a = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        policy_b = BlueprintPolicy(iteration=2, preflop_average={}, postflop_current={})

        result = run_one_vs_field_league(
            policies={"alpha": policy_a, "beta": policy_b},
            config=LeagueEvaluationConfig(
                num_hands_per_seat=2,
                random_seed=17,
                sample_actions=False,
            ),
        )

        self.assertEqual(result["format"], "one_vs_field")
        self.assertEqual(result["num_players"], 6)
        self.assertEqual(result["total_hands_per_matchup"], 12)
        self.assertEqual(len(result["matchups"]), 2)

        matchup_keys = {(m["candidate"], m["field"]) for m in result["matchups"]}
        self.assertEqual(matchup_keys, {("alpha", "beta"), ("beta", "alpha")})

        for matchup in result["matchups"]:
            self.assertEqual(matchup["hands_played"], 12)
            self.assertAlmostEqual(float(matchup["zero_sum_max_abs_error"]), 0.0, places=6)
            self.assertIn("ci95_mbb_per_hand", matchup)
            self.assertEqual(len(matchup["ci95_mbb_per_hand"]), 2)

        self.assertEqual(set(result["matrix_mbb_per_hand"].keys()), {"alpha", "beta"})
        self.assertIn("beta", result["matrix_mbb_per_hand"]["alpha"])
        self.assertIn("alpha", result["matrix_mbb_per_hand"]["beta"])
        self.assertEqual(set(result["aggregates"].keys()), {"alpha", "beta"})
        self.assertEqual(result["aggregates"]["alpha"]["hands_played"], 12)
        self.assertEqual(result["aggregates"]["beta"]["hands_played"], 12)

    def test_load_blueprint_policy_from_json_file(self) -> None:
        payload = {
            "iteration": 42,
            "preflop_average": {"k1": [0.1, 0.9]},
            "postflop_current": {"k2": [0.2, 0.8]},
        }

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "blueprint.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            policy = load_blueprint_policy(path)

        self.assertEqual(policy.iteration, 42)
        self.assertIn("k1", policy.preflop_average)
        self.assertIn("k2", policy.postflop_current)

    def test_load_blueprint_policy_reuses_payload_cache(self) -> None:
        payload = {
            "iteration": 7,
            "preflop_average": {"k1": [0.1, 0.9]},
            "postflop_current": {"k2": [0.2, 0.8]},
        }

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "blueprint.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)

            first = load_blueprint_policy(path)
            cache_path = Path(f"{path}.cache.pkl")
            self.assertTrue(cache_path.exists())

            with patch("pluribus_ri.blueprint.evaluation.json.load", side_effect=RuntimeError("json.load called")):
                second = load_blueprint_policy(path)

        self.assertEqual(first.iteration, 7)
        self.assertEqual(second.iteration, 7)

    def test_load_blueprint_policy_invalidates_cache_on_source_change(self) -> None:
        payload_v1 = {
            "iteration": 11,
            "preflop_average": {"k1": [0.1, 0.9]},
            "postflop_current": {"k2": [0.2, 0.8]},
        }
        payload_v2 = {
            "iteration": 99,
            "preflop_average": {"k1": [0.2, 0.8], "k3": [1.0, 0.0]},
            "postflop_current": {"k2": [0.9, 0.1], "k4": [0.5, 0.5]},
        }

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "blueprint.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload_v1, f)

            first = load_blueprint_policy(path)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload_v2, f)
            second = load_blueprint_policy(path)

        self.assertEqual(first.iteration, 11)
        self.assertEqual(second.iteration, 99)

    def test_league_requires_at_least_two_policies(self) -> None:
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        with self.assertRaises(ValueError):
            run_one_vs_field_league(
                policies={"solo": policy},
                config=LeagueEvaluationConfig(num_hands_per_seat=1),
            )

    def test_exploitability_proxy_report_supports_single_candidate(self) -> None:
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        report = run_exploitability_proxy_report(
            policies={"alpha": policy},
            config=ExploitabilityProxyConfig(
                num_hands_per_seat=1,
                random_seed=5,
                sample_actions=False,
                baseline_policies=("check_fold", "call_biased"),
            ),
        )

        self.assertEqual(report["format"], "exploitability_proxy")
        self.assertEqual(report["baseline_policies"], ["check_fold", "call_biased"])
        self.assertEqual(len(report["candidates"]), 1)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["candidate"], "alpha")
        self.assertEqual(len(candidate["baseline_matchups"]), 2)
        self.assertGreaterEqual(float(candidate["proxy_exploitability_mbb_per_hand"]), 0.0)
        for baseline_matchup in candidate["baseline_matchups"]:
            self.assertAlmostEqual(float(baseline_matchup["zero_sum_max_abs_error"]), 0.0, places=6)

    def test_exploitability_proxy_supports_strength_heuristic_baselines(self) -> None:
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        report = run_exploitability_proxy_report(
            policies={"alpha": policy},
            config=ExploitabilityProxyConfig(
                num_hands_per_seat=1,
                random_seed=7,
                sample_actions=True,
                baseline_policies=("tight_aggressive", "loose_aggressive", "pot_odds"),
            ),
        )

        candidate = report["candidates"][0]
        observed = [item["baseline"] for item in candidate["baseline_matchups"]]
        self.assertEqual(observed, ["tight_aggressive", "loose_aggressive", "pot_odds"])
        self.assertGreaterEqual(float(candidate["proxy_exploitability_mbb_per_hand"]), 0.0)
        for baseline_matchup in candidate["baseline_matchups"]:
            self.assertAlmostEqual(float(baseline_matchup["zero_sum_max_abs_error"]), 0.0, places=6)

    def test_exploitability_proxy_supports_cheater_baselines(self) -> None:
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        report = run_exploitability_proxy_report(
            policies={"alpha": policy},
            config=ExploitabilityProxyConfig(
                num_hands_per_seat=1,
                random_seed=11,
                sample_actions=True,
                baseline_policies=("cheater_weak", "cheater_strong"),
            ),
        )

        candidate = report["candidates"][0]
        observed = [item["baseline"] for item in candidate["baseline_matchups"]]
        self.assertEqual(observed, ["cheater_weak", "cheater_strong"])
        self.assertGreaterEqual(float(candidate["proxy_exploitability_mbb_per_hand"]), 0.0)
        for baseline_matchup in candidate["baseline_matchups"]:
            self.assertAlmostEqual(float(baseline_matchup["zero_sum_max_abs_error"]), 0.0, places=6)

    def test_exploitability_proxy_rejects_unknown_baseline(self) -> None:
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        invalid = tuple(list(SUPPORTED_BASELINES) + ["unknown"])
        with self.assertRaises(ValueError):
            run_exploitability_proxy_report(
                policies={"alpha": policy},
                config=ExploitabilityProxyConfig(
                    num_hands_per_seat=1,
                    baseline_policies=invalid,  # type: ignore[arg-type]
                ),
            )

    def test_league_control_variate_summary_is_reported(self) -> None:
        policy_a = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        policy_b = BlueprintPolicy(iteration=2, preflop_average={}, postflop_current={})
        result = run_one_vs_field_league(
            policies={"alpha": policy_a, "beta": policy_b},
            config=LeagueEvaluationConfig(
                num_hands_per_seat=2,
                random_seed=23,
                sample_actions=False,
                control_variate_baseline="uniform",
            ),
        )

        for matchup in result["matchups"]:
            self.assertIn("control_variate", matchup)
            control_variate = matchup["control_variate"]
            self.assertEqual(control_variate["baseline_strategy"], "uniform")
            self.assertAlmostEqual(
                float(control_variate["adjusted_mean_utility_per_hand"]),
                float(matchup["mean_utility_per_hand"]),
                places=6,
            )
            self.assertIn("ci95_width_reduction_pct", control_variate)

    def test_proxy_control_variate_summary_is_reported(self) -> None:
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        report = run_exploitability_proxy_report(
            policies={"alpha": policy},
            config=ExploitabilityProxyConfig(
                num_hands_per_seat=2,
                random_seed=29,
                sample_actions=False,
                baseline_policies=("check_fold",),
                control_variate_baseline="uniform",
            ),
        )

        baseline_matchup = report["candidates"][0]["baseline_matchups"][0]
        self.assertIn("control_variate", baseline_matchup)
        control_variate = baseline_matchup["control_variate"]
        self.assertEqual(control_variate["baseline_strategy"], "uniform")
        self.assertAlmostEqual(
            float(control_variate["adjusted_mean_utility_per_hand"]),
            float(baseline_matchup["mean_utility_per_hand"]),
            places=6,
        )

    def test_control_variate_rejects_unknown_baseline(self) -> None:
        policy_a = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        policy_b = BlueprintPolicy(iteration=2, preflop_average={}, postflop_current={})
        with self.assertRaises(ValueError):
            run_one_vs_field_league(
                policies={"alpha": policy_a, "beta": policy_b},
                config=LeagueEvaluationConfig(
                    num_hands_per_seat=1,
                    control_variate_baseline="unknown",  # type: ignore[arg-type]
                ),
            )

    def test_league_aivat_summary_is_reported(self) -> None:
        policy_a = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        policy_b = BlueprintPolicy(iteration=2, preflop_average={}, postflop_current={})
        result = run_one_vs_field_league(
            policies={"alpha": policy_a, "beta": policy_b},
            config=LeagueEvaluationConfig(
                num_hands_per_seat=2,
                random_seed=31,
                sample_actions=True,
                aivat_config=AIVATConfig(
                    rollout_count_per_action=1,
                    max_actions_per_rollout=32,
                    max_branching_for_correction=8,
                    include_opponent_decisions=True,
                ),
            ),
        )

        self.assertIsNotNone(result["aivat_config"])
        for matchup in result["matchups"]:
            self.assertIn("aivat", matchup)
            aivat = matchup["aivat"]
            self.assertEqual(aivat["method"], "action_correction_only")
            self.assertIn("adjusted_ci95_mbb_per_hand", aivat)
            self.assertIn("ci95_width_reduction_pct", aivat)
            self.assertGreaterEqual(int(aivat["corrections_applied"]), 0)

    def test_proxy_aivat_summary_is_reported(self) -> None:
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        report = run_exploitability_proxy_report(
            policies={"alpha": policy},
            config=ExploitabilityProxyConfig(
                num_hands_per_seat=2,
                random_seed=41,
                sample_actions=True,
                baseline_policies=("uniform",),
                aivat_config=AIVATConfig(
                    rollout_count_per_action=1,
                    max_actions_per_rollout=24,
                    max_branching_for_correction=8,
                    include_opponent_decisions=False,
                ),
            ),
        )
        self.assertIsNotNone(report["aivat_config"])
        matchup = report["candidates"][0]["baseline_matchups"][0]
        self.assertIn("aivat", matchup)
        self.assertIn("adjusted_mean_utility_per_hand", matchup["aivat"])

    def test_aivat_rejects_invalid_rollout_count(self) -> None:
        policy_a = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        policy_b = BlueprintPolicy(iteration=2, preflop_average={}, postflop_current={})
        with self.assertRaises(ValueError):
            run_one_vs_field_league(
                policies={"alpha": policy_a, "beta": policy_b},
                config=LeagueEvaluationConfig(
                    num_hands_per_seat=1,
                    aivat_config=AIVATConfig(
                        rollout_count_per_action=0,
                        max_actions_per_rollout=16,
                        max_branching_for_correction=4,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
