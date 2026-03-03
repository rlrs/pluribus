import unittest

from pluribus_ri.abstraction.state_indexer import (
    build_public_state_key,
    encode_engine_infoset_key,
    private_hand_bucket_with_policy,
)
from pluribus_ri.abstraction import state_indexer_kernels
from pluribus_ri.core import Action, NoLimitHoldemEngine


class StateIndexerKernelParityTests(unittest.TestCase):
    def _engine_with_actions(self) -> NoLimitHoldemEngine:
        engine = NoLimitHoldemEngine(seed=0)
        engine.reset_match(stacks=[10_000] * 6, button=0)
        engine.start_hand()

        for step in range(4):
            legal = engine.get_legal_actions()
            if step == 0 and legal.min_raise_to is not None:
                engine.apply_action(Action(kind="raise", amount=legal.min_raise_to))
                continue
            if legal.call_amount > 0:
                engine.apply_action(Action(kind="call"))
                continue
            if legal.can_check:
                engine.apply_action(Action(kind="check"))
                continue
            engine.apply_action(Action(kind="fold"))

            if engine.hand_complete:
                break

        return engine

    def test_public_state_token_matches_reference(self) -> None:
        engine = self._engine_with_actions()
        for scope in ("street", "all"):
            with self.subTest(scope=scope):
                got = state_indexer_kernels.build_public_state_token(engine, scope)
                expected = build_public_state_key(engine, history_scope=scope).to_token()
                self.assertEqual(got, expected)

    def test_infoset_key_matches_reference_composition(self) -> None:
        engine = self._engine_with_actions()
        seat = int(engine.to_act)
        for scope in ("street", "all"):
            with self.subTest(scope=scope):
                got = encode_engine_infoset_key(engine, seat=seat, history_scope=scope)
                bucket = private_hand_bucket_with_policy(engine, seat=seat)
                expected = f"p{seat}|b{bucket}|{build_public_state_key(engine, history_scope=scope).to_token()}"
                self.assertEqual(got, expected)

    def test_loader_exports_flag(self) -> None:
        self.assertIsInstance(state_indexer_kernels.USING_CYTHON_STATE_INDEXER_KERNELS, bool)


if __name__ == "__main__":
    unittest.main()
