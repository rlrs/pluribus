import unittest

from pluribus_ri.abstraction import action_kernels
from pluribus_ri.abstraction import _py_action_kernels
from pluribus_ri.abstraction.game_builder import ActionAbstractionConfig
from pluribus_ri.core import Action, NoLimitHoldemEngine


class ActionKernelParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ActionAbstractionConfig(
            preflop_raise_multipliers=(2.0, 3.0, 5.0, 10.0),
            postflop_pot_raise_fractions=(0.5, 1.0, 2.0),
            max_raise_actions=4,
        )

    def _engine_preflop(self) -> NoLimitHoldemEngine:
        engine = NoLimitHoldemEngine(seed=0)
        engine.reset_match(stacks=[10_000] * 6, button=0)
        engine.start_hand()
        return engine

    def _engine_flop(self) -> NoLimitHoldemEngine:
        engine = self._engine_preflop()
        while not engine.hand_complete and engine.street.value == "preflop":
            legal = engine.get_legal_actions()
            if legal.call_amount > 0:
                engine.apply_action(Action(kind="call"))
            else:
                engine.apply_action(Action(kind="check"))
        return engine

    def test_preflop_actions_match_reference(self) -> None:
        engine = self._engine_preflop()
        got = action_kernels.legal_action_specs(engine, self.config)
        expected = _py_action_kernels.legal_action_specs(engine, self.config)
        self.assertEqual(got, expected)

    def test_flop_actions_match_reference(self) -> None:
        engine = self._engine_flop()
        if engine.hand_complete:
            self.skipTest("hand completed before flop in deterministic rollout")
        got = action_kernels.legal_action_specs(engine, self.config)
        expected = _py_action_kernels.legal_action_specs(engine, self.config)
        self.assertEqual(got, expected)

    def test_loader_exports_flag(self) -> None:
        self.assertIsInstance(action_kernels.USING_CYTHON_ACTION_KERNELS, bool)


if __name__ == "__main__":
    unittest.main()
