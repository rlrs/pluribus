from dataclasses import dataclass
import unittest

from pluribus_ri.solver import ExternalSamplingLinearMCCFR, LazyIntRegretTable, MCCFRConfig


@dataclass(frozen=True)
class KuhnState:
    cards: tuple[int, int] | None = None
    history: str = ""

    def is_terminal(self) -> bool:
        return self.history in {"cc", "bc", "bf", "cbc", "cbf"}

    def utility(self, player: int) -> float:
        if not self.is_terminal() or self.cards is None:
            raise ValueError("utility requested for non-terminal state")

        contributions = [1, 1]

        if self.history in {"bf", "bc"}:
            contributions[0] += 1
            if self.history == "bc":
                contributions[1] += 1
        elif self.history in {"cbf", "cbc"}:
            contributions[1] += 1
            if self.history == "cbc":
                contributions[0] += 1

        if self.history == "bf":
            winner = 0
        elif self.history == "cbf":
            winner = 1
        else:
            winner = 0 if self.cards[0] > self.cards[1] else 1

        pot = sum(contributions)
        if player == winner:
            return float(pot - contributions[player])
        return float(-contributions[player])

    def is_chance_node(self) -> bool:
        return self.cards is None

    def chance_outcomes(self):
        if not self.is_chance_node():
            raise ValueError("chance outcomes requested for non-chance state")
        outcomes = []
        for c0 in (0, 1, 2):
            for c1 in (0, 1, 2):
                if c0 == c1:
                    continue
                outcomes.append(((c0, c1), 1.0 / 6.0))
        return outcomes

    def current_player(self) -> int:
        if self.is_chance_node() or self.is_terminal():
            raise ValueError("current_player requested outside decision node")
        if self.history == "":
            return 0
        if self.history == "c":
            return 1
        if self.history == "b":
            return 1
        if self.history == "cb":
            return 0
        raise ValueError(f"unsupported history {self.history}")

    def legal_actions(self):
        if self.is_chance_node() or self.is_terminal():
            return []
        if self.history == "":
            return ["c", "b"]
        if self.history == "c":
            return ["c", "b"]
        if self.history == "b":
            return ["f", "c"]
        if self.history == "cb":
            return ["f", "c"]
        raise ValueError(f"unsupported history {self.history}")

    def child(self, action):
        if self.is_chance_node():
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError("chance action must be (card0, card1)")
            return KuhnState(cards=(int(action[0]), int(action[1])), history=self.history)

        if self.history == "":
            if action == "c":
                return KuhnState(cards=self.cards, history="c")
            if action == "b":
                return KuhnState(cards=self.cards, history="b")
        elif self.history == "c":
            if action == "c":
                return KuhnState(cards=self.cards, history="cc")
            if action == "b":
                return KuhnState(cards=self.cards, history="cb")
        elif self.history == "b":
            if action == "f":
                return KuhnState(cards=self.cards, history="bf")
            if action == "c":
                return KuhnState(cards=self.cards, history="bc")
        elif self.history == "cb":
            if action == "f":
                return KuhnState(cards=self.cards, history="cbf")
            if action == "c":
                return KuhnState(cards=self.cards, history="cbc")

        raise ValueError(f"illegal action {action} in history {self.history}")

    def infoset_key(self, player: int) -> str:
        if self.cards is None:
            raise ValueError("infoset key requested before cards are dealt")
        return f"kuhn:p{player}:card={self.cards[player]}:h={self.history}"


class SolverScaffoldTests(unittest.TestCase):
    def test_lazy_regret_table_floor_and_strategy(self) -> None:
        table = LazyIntRegretTable(regret_floor=-10)

        table.add_regret(key="k", action_index=0, delta=-100.0, num_actions=2)
        table.add_regret(key="k", action_index=1, delta=5.0, num_actions=2)

        self.assertEqual(table.get_regrets("k", 2), [-10, 5])

        current = table.current_strategy("k", 2)
        self.assertAlmostEqual(current[0], 0.0)
        self.assertAlmostEqual(current[1], 1.0)

    def test_lazy_regret_table_serialize_roundtrip(self) -> None:
        table = LazyIntRegretTable(regret_floor=-123)
        table.add_regret("x", action_index=0, delta=4.0, num_actions=2)
        table.add_regret("x", action_index=1, delta=-2.0, num_actions=2)
        table.accumulate_average_strategy("x", [0.7, 0.3], num_actions=2, weight=5.0)

        restored = LazyIntRegretTable.deserialize(table.serialize())
        self.assertEqual(restored.regret_floor, -123)
        self.assertEqual(restored.get_regrets("x", 2), table.get_regrets("x", 2))
        self.assertEqual(
            restored.get_average_strategy_sums("x", 2),
            table.get_average_strategy_sums("x", 2),
        )

    def test_external_sampling_linear_mccfr_runs_and_is_reproducible(self) -> None:
        def train_snapshot() -> dict[str, list[int]]:
            table = LazyIntRegretTable()
            trainer = ExternalSamplingLinearMCCFR(
                regret_table=table,
                config=MCCFRConfig(
                    iterations=150,
                    random_seed=11,
                    prune_after_iteration=75,
                    negative_regret_pruning_threshold=-100,
                ),
            )
            stats = trainer.train(root_state_factory=KuhnState, num_players=2)
            self.assertEqual(stats.iterations_completed, 150)
            self.assertEqual(stats.traversals_completed, 300)
            self.assertGreater(stats.nodes_touched, 0)
            self.assertGreater(table.infoset_count, 0)

            for key in table.keys():
                num_actions = table.num_actions(key)
                avg = table.average_strategy(key, num_actions)
                self.assertAlmostEqual(sum(avg), 1.0, places=6)

            return table.snapshot_regrets()

        snapshot_a = train_snapshot()
        snapshot_b = train_snapshot()
        self.assertEqual(snapshot_a, snapshot_b)

    def test_train_steps_supports_incremental_runs(self) -> None:
        table = LazyIntRegretTable()
        trainer = ExternalSamplingLinearMCCFR(
            regret_table=table,
            config=MCCFRConfig(iterations=10, random_seed=7),
        )

        callback_iterations: list[int] = []
        trainer.train_steps(
            root_state_factory=KuhnState,
            num_players=2,
            iterations=3,
            on_iteration_end=lambda iteration, _: callback_iterations.append(iteration),
        )
        self.assertEqual(trainer.stats.iterations_completed, 3)
        self.assertEqual(callback_iterations, [1, 2, 3])

        trainer.train_steps(
            root_state_factory=KuhnState,
            num_players=2,
            iterations=2,
        )
        self.assertEqual(trainer.stats.iterations_completed, 5)
        self.assertEqual(trainer.stats.traversals_completed, 10)


if __name__ == "__main__":
    unittest.main()
