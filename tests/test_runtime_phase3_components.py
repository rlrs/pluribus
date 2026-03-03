import unittest

from pluribus_ri.solver import NLTHAbstractAction
from pluribus_ri.runtime_search import (
    LeafContinuationConfig,
    SubgameSearchStopController,
    SubgameStoppingRules,
    apply_continuation_bias,
)


class RuntimePhase3ComponentTests(unittest.TestCase):
    def test_apply_continuation_bias(self) -> None:
        actions = [
            NLTHAbstractAction(kind="fold"),
            NLTHAbstractAction(kind="call"),
            NLTHAbstractAction(kind="raise", amount=500),
        ]
        base = [0.2, 0.3, 0.5]

        raise_biased = apply_continuation_bias(actions, base, "raise_biased")
        fold_biased = apply_continuation_bias(actions, base, "fold_biased")
        call_biased = apply_continuation_bias(actions, base, "call_biased")

        self.assertAlmostEqual(sum(raise_biased), 1.0, places=6)
        self.assertAlmostEqual(sum(fold_biased), 1.0, places=6)
        self.assertAlmostEqual(sum(call_biased), 1.0, places=6)

        self.assertGreater(raise_biased[2], base[2])
        self.assertGreater(fold_biased[0], base[0])
        self.assertGreater(call_biased[1], base[1])

    def test_stop_controller_respects_min_then_limits(self) -> None:
        controller = SubgameSearchStopController(
            SubgameStoppingRules(
                min_cfr_iterations=2,
                max_cfr_iterations=3,
                max_nodes_touched=100,
                max_wallclock_ms=10_000,
                leaf_max_depth=8,
            )
        )

        should_stop, reason = controller.should_stop(iterations=1, nodes_touched=200)
        self.assertFalse(should_stop)
        self.assertEqual(reason, "min_iterations_not_reached")

        should_stop, reason = controller.should_stop(iterations=3, nodes_touched=10)
        self.assertTrue(should_stop)
        self.assertEqual(reason, "max_iterations")

    def test_leaf_continuation_config_defaults(self) -> None:
        config = LeafContinuationConfig()
        self.assertGreater(config.rollout_count, 0)
        self.assertGreater(config.max_actions_per_rollout, 0)
        self.assertEqual(len(config.strategy_mix), 4)


if __name__ == "__main__":
    unittest.main()
