import copy
from dataclasses import dataclass
import random
import unittest

from pluribus_ri.core import Action, NoLimitHoldemEngine
from pluribus_ri.solver import ExternalSamplingLinearMCCFR, LazyIntRegretTable, MCCFRConfig


@dataclass(frozen=True)
class SingleDecisionState:
    terminal: bool = False
    chosen_action: int | None = None

    def is_terminal(self) -> bool:
        return self.terminal

    def utility(self, player: int) -> float:
        if not self.terminal or self.chosen_action is None:
            raise ValueError("utility requested for non-terminal state")
        payoff_player_0 = 1.0 if self.chosen_action == 0 else -1.0
        return payoff_player_0 if player == 0 else -payoff_player_0

    def is_chance_node(self) -> bool:
        return False

    def chance_outcomes(self):
        return []

    def current_player(self) -> int:
        if self.terminal:
            raise ValueError("no current player in terminal state")
        return 0

    def legal_actions(self):
        return [] if self.terminal else [0, 1]

    def child(self, action):
        if self.terminal:
            raise ValueError("cannot act in terminal state")
        if action not in (0, 1):
            raise ValueError("invalid action")
        return SingleDecisionState(terminal=True, chosen_action=int(action))

    def infoset_key(self, player: int) -> str:
        return f"single-state:p{player}"


class EngineCoreVerificationTests(unittest.TestCase):
    def test_randomized_engine_invariants_and_legal_action_soundness(self) -> None:
        rng = random.Random(12345)
        engine = NoLimitHoldemEngine(seed=2026)

        total_chips = sum(engine.stacks)

        hands_completed = 0
        for _ in range(30):
            if sum(1 for stack in engine.stacks if stack > 0) < 2:
                break

            engine.start_hand()
            hands_completed += 1

            step_guard = 0
            while not engine.hand_complete:
                step_guard += 1
                self.assertLess(step_guard, 500)

                legal = engine.get_legal_actions()

                self._assert_legal_actions_apply(engine, legal)
                self._assert_illegal_actions_rejected(engine, legal)

                selected = self._pick_random_legal_action(engine, legal, rng)
                engine.apply_action(selected)

                if not engine.hand_complete:
                    self.assertEqual(
                        sum(player.stack + player.contributed_total for player in engine.players),
                        total_chips,
                    )
                else:
                    self.assertEqual(sum(engine.stacks), total_chips)

            self.assertEqual(sum(engine.stacks), total_chips)

        self.assertGreater(hands_completed, 0)

    def _pick_random_legal_action(self, engine: NoLimitHoldemEngine, legal, rng: random.Random) -> Action:
        options: list[Action] = []
        if legal.can_fold:
            options.append(Action(kind="fold"))

        if legal.can_check:
            options.append(Action(kind="check"))
        elif legal.call_amount > 0:
            options.append(Action(kind="call"))

        if legal.min_raise_to is not None and legal.max_raise_to is not None:
            raise_to = (
                legal.min_raise_to
                if legal.min_raise_to == legal.max_raise_to
                else rng.randint(legal.min_raise_to, legal.max_raise_to)
            )
            options.append(Action(kind="raise", amount=raise_to))

        if not options:
            self.fail("no legal action options generated")

        return rng.choice(options)

    def _assert_legal_actions_apply(self, engine: NoLimitHoldemEngine, legal) -> None:
        if legal.can_fold:
            folded = copy.deepcopy(engine)
            folded.apply_action(Action(kind="fold"))

        if legal.can_check:
            checked = copy.deepcopy(engine)
            checked.apply_action(Action(kind="check"))
        elif legal.call_amount > 0:
            called = copy.deepcopy(engine)
            called.apply_action(Action(kind="call"))

        if legal.min_raise_to is not None and legal.max_raise_to is not None:
            raised = copy.deepcopy(engine)
            raised.apply_action(Action(kind="raise", amount=legal.min_raise_to))

    def _assert_illegal_actions_rejected(self, engine: NoLimitHoldemEngine, legal) -> None:
        if not legal.can_check:
            with self.assertRaises(ValueError):
                copy.deepcopy(engine).apply_action(Action(kind="check"))

        if legal.call_amount <= 0:
            with self.assertRaises(ValueError):
                copy.deepcopy(engine).apply_action(Action(kind="call"))

        if legal.min_raise_to is None or legal.max_raise_to is None:
            with self.assertRaises(ValueError):
                copy.deepcopy(engine).apply_action(Action(kind="raise", amount=engine.current_bet + 1))
            return

        with self.assertRaises(ValueError):
            copy.deepcopy(engine).apply_action(Action(kind="raise", amount=legal.min_raise_to - 1))

        with self.assertRaises(ValueError):
            copy.deepcopy(engine).apply_action(Action(kind="raise", amount=legal.max_raise_to + 1))


class SolverCoreVerificationTests(unittest.TestCase):
    def test_solver_prefers_better_action_in_single_decision_game(self) -> None:
        table = LazyIntRegretTable()
        trainer = ExternalSamplingLinearMCCFR(
            regret_table=table,
            config=MCCFRConfig(iterations=200, random_seed=11),
        )

        trainer.train(root_state_factory=SingleDecisionState, num_players=2)

        key = "single-state:p0"
        regrets = table.get_regrets(key, 2)
        current = table.current_strategy(key, 2)
        average = table.average_strategy(key, 2)

        self.assertGreater(regrets[0], regrets[1])
        self.assertGreater(current[0], 0.95)
        self.assertGreater(average[0], 0.70)


if __name__ == "__main__":
    unittest.main()
