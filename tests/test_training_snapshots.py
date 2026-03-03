import unittest

from pluribus_ri.solver import LazyIntRegretTable
from pluribus_ri.training import build_strategy_snapshot, extract_street_from_infoset_key


class StrategySnapshotTests(unittest.TestCase):
    def test_extract_street_from_infoset_key(self) -> None:
        key = "p0|b12|street=flop|board=AhKd2c|to_act=3"
        self.assertEqual(extract_street_from_infoset_key(key), "flop")
        self.assertIsNone(extract_street_from_infoset_key("p0|b12|board=AhKd2c"))

    def test_snapshot_collects_preflop_average_and_postflop_current(self) -> None:
        table = LazyIntRegretTable()

        preflop_key = "p0|b1|street=preflop|board=|to_act=3"
        flop_key = "p1|b2|street=flop|board=AhKd2c|to_act=1"

        table.ensure_infoset(preflop_key, 2)
        table.ensure_infoset(flop_key, 2)

        table.accumulate_average_strategy(preflop_key, [0.8, 0.2], 2, weight=10.0)
        table.accumulate_average_strategy(preflop_key, [0.4, 0.6], 2, weight=5.0)

        table.add_regret(flop_key, 0, delta=0.0, num_actions=2)
        table.add_regret(flop_key, 1, delta=3.0, num_actions=2)

        snapshot = build_strategy_snapshot(table=table, iteration=42)
        self.assertEqual(snapshot.iteration, 42)

        self.assertIn(preflop_key, snapshot.preflop_average)
        self.assertNotIn(preflop_key, snapshot.postflop_current)

        self.assertIn(flop_key, snapshot.postflop_current)
        self.assertNotIn(flop_key, snapshot.preflop_average)

        preflop_probs = snapshot.preflop_average[preflop_key]
        self.assertAlmostEqual(sum(preflop_probs), 1.0)

        flop_probs = snapshot.postflop_current[flop_key]
        self.assertAlmostEqual(flop_probs[0], 0.0)
        self.assertAlmostEqual(flop_probs[1], 1.0)


if __name__ == "__main__":
    unittest.main()
