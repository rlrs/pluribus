import json
from pathlib import Path
import tempfile
import unittest

from pluribus_ri.training import (
    Phase1RunConfig,
    Phase2RunConfig,
    run_phase1_training,
    run_phase2_training,
)


class Phase1TrainingRunnerTests(unittest.TestCase):
    def test_runner_writes_checkpoints_snapshots_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "phase1"

            summary = run_phase1_training(
                Phase1RunConfig(
                    output_dir=str(output_dir),
                    iterations=2,
                    checkpoint_interval=1,
                    snapshot_interval=1,
                    random_seed=5,
                    max_raise_actions=2,
                )
            )

            self.assertEqual(summary["iterations_completed"], 2)
            self.assertEqual(len(summary["checkpoints"]), 2)
            self.assertEqual(len(summary["snapshots"]), 2)
            self.assertGreater(summary["infosets_allocated"], 0)

            summary_path = Path(summary["summary_path"])
            self.assertTrue(summary_path.exists())

            checkpoint_path = output_dir / "checkpoints" / "checkpoint_iter_000002.json"
            snapshot_path = output_dir / "snapshots" / "snapshot_iter_000002.json"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(snapshot_path.exists())

            checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_data["iteration"], 2)
            self.assertIn("regret_table", checkpoint_data)

            snapshot_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot_data["iteration"], 2)
            self.assertIn("preflop_average", snapshot_data)
            self.assertIn("postflop_current", snapshot_data)

    def test_runner_supports_phase4_abstraction_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "phase1_phase4_controls"

            summary = run_phase1_training(
                Phase1RunConfig(
                    output_dir=str(output_dir),
                    iterations=1,
                    checkpoint_interval=1,
                    snapshot_interval=1,
                    random_seed=7,
                    preflop_raise_multipliers=(2.0, 2.5, 4.0),
                    postflop_pot_raise_fractions=(0.5, 1.0),
                    flop_pot_raise_fractions=(0.25, 0.75, 1.25),
                    turn_pot_raise_fractions=(0.5, 1.0),
                    river_pot_raise_fractions=(0.5,),
                    preflop_bucket_policy="canonical169",
                    postflop_bucket_policy="texture_v1",
                    max_raise_actions=3,
                )
            )

            self.assertEqual(summary["iterations_completed"], 1)
            self.assertGreater(summary["infosets_allocated"], 0)


class Phase2TrainingRunnerTests(unittest.TestCase):
    def test_phase2_runner_writes_playable_blueprint_and_self_play_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "phase2"

            summary = run_phase2_training(
                Phase2RunConfig(
                    output_dir=str(output_dir),
                    iterations=2,
                    checkpoint_interval=1,
                    snapshot_interval=1,
                    random_seed=13,
                    max_raise_actions=2,
                    self_play_hands=4,
                    self_play_seed=77,
                )
            )

            self.assertEqual(summary["phase"], "phase_2")
            self.assertEqual(summary["iterations_completed"], 2)
            self.assertIn("blueprint_policy_path", summary)
            self.assertIn("self_play", summary)

            blueprint_path = Path(summary["blueprint_policy_path"])
            self.assertTrue(blueprint_path.exists())

            blueprint_data = json.loads(blueprint_path.read_text(encoding="utf-8"))
            self.assertEqual(blueprint_data["iteration"], 2)
            self.assertIn("preflop_average", blueprint_data)
            self.assertIn("postflop_current", blueprint_data)

            self_play = summary["self_play"]
            self.assertEqual(self_play["hands_played"], 4)
            self.assertEqual(len(self_play["utility_sums"]), 6)
            self.assertAlmostEqual(float(self_play["zero_sum_check"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
