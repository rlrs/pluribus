import unittest

from pluribus_ri.solver import (
    NLTHAbstractAction,
    NLTHAbstractGameFactory,
    NLTHActionAbstractionConfig,
    NLTHGameConfig,
)


class NLTHGameAdapterTests(unittest.TestCase):
    def test_preflop_abstract_actions_include_core_options(self) -> None:
        factory = NLTHAbstractGameFactory(
            game_config=NLTHGameConfig(random_seed=9),
            abstraction_config=NLTHActionAbstractionConfig(
                preflop_raise_multipliers=(2.0, 3.0, 5.0, 10.0),
                max_raise_actions=3,
            ),
        )

        state = factory.root_state()
        self.assertFalse(state.is_terminal())
        self.assertEqual(state.current_player(), 3)

        actions = list(state.legal_actions())
        kinds = [a.kind for a in actions]
        raise_actions = [a for a in actions if a.kind == "raise"]

        self.assertIn("fold", kinds)
        self.assertIn("call", kinds)
        self.assertGreaterEqual(len(raise_actions), 1)
        self.assertLessEqual(len(raise_actions), 3)

        # Ensure deduplicated raise targets.
        raise_targets = [a.amount for a in raise_actions]
        self.assertEqual(len(raise_targets), len(set(raise_targets)))

    def test_child_transition_infoset_and_terminal_utility_zero_sum(self) -> None:
        factory = NLTHAbstractGameFactory(game_config=NLTHGameConfig(random_seed=3))
        state = factory.root_state()

        seat = state.current_player()
        infoset = state.infoset_key(seat)
        self.assertTrue(infoset.startswith(f"p{seat}|b"))
        self.assertIn("street=preflop", infoset)

        # Apply a legal non-raise action and verify turn advances.
        non_raise = next(action for action in state.legal_actions() if action.kind in {"call", "check", "fold"})
        next_state = state.child(non_raise)
        self.assertNotEqual(next_state.current_player(), seat)

        # Drive to terminal by folding whenever possible.
        current = next_state
        guard = 0
        while not current.is_terminal():
            guard += 1
            self.assertLess(guard, 200)
            legal = list(current.legal_actions())
            action = _pick_fast_terminal_action(legal)
            current = current.child(action)

        utilities = [current.utility(player) for player in range(6)]
        self.assertAlmostEqual(sum(utilities), 0.0, places=6)



def _pick_fast_terminal_action(actions: list[NLTHAbstractAction]) -> NLTHAbstractAction:
    for kind in ("fold", "check", "call", "raise"):
        for action in actions:
            if action.kind == kind:
                return action
    raise ValueError("no legal action available")


if __name__ == "__main__":
    unittest.main()
