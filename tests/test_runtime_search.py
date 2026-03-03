import unittest

from pluribus_ri.core import Action, NoLimitHoldemEngine, Street
from pluribus_ri.runtime_search import (
    ActionLikelihoodModel,
    OutsideObserverBeliefState,
    all_private_hand_combos,
    build_public_search_root,
)


class _PairPreferringRaiseModel(ActionLikelihoodModel):
    def likelihood(
        self,
        seat: int,
        hand: tuple[str, str],
        action: Action,
        engine: NoLimitHoldemEngine,
    ) -> float:
        del seat, engine
        if action.kind != "raise":
            return 1.0
        return 4.0 if hand[0][0] == hand[1][0] else 1.0


class RuntimeSearchScaffoldTests(unittest.TestCase):
    def test_private_hand_combo_counts(self) -> None:
        preflop = all_private_hand_combos()
        self.assertEqual(len(preflop), 1326)

        postflop = all_private_hand_combos(excluded_cards=("Ah", "Kd", "2c"))
        self.assertEqual(len(postflop), 1176)

    def test_beliefs_initialize_from_public_state(self) -> None:
        engine = NoLimitHoldemEngine(seed=202)
        engine.start_hand()
        _advance_to_flop(engine)

        beliefs = OutsideObserverBeliefState.from_engine_public_state(engine)
        self.assertEqual(tuple(engine.board), beliefs.public_cards)

        seat_belief = beliefs.players[0]
        self.assertEqual(len(seat_belief.probs), 1176)
        self.assertAlmostEqual(sum(seat_belief.probs.values()), 1.0, places=8)

    def test_belief_update_reweights_toward_pairs(self) -> None:
        engine = NoLimitHoldemEngine(seed=303)
        engine.start_hand()

        seat = int(engine.to_act)  # type: ignore[arg-type]
        legal = engine.get_legal_actions(seat)
        self.assertIsNotNone(legal.min_raise_to)

        beliefs = OutsideObserverBeliefState.from_engine_public_state(engine)
        prior = _pair_probability(beliefs.players[seat].probs)

        beliefs.observe_action(
            seat=seat,
            action=Action(kind="raise", amount=int(legal.min_raise_to)),  # type: ignore[arg-type]
            engine=engine,
            likelihood_model=_PairPreferringRaiseModel(),
        )
        posterior = _pair_probability(beliefs.players[seat].probs)

        self.assertGreater(posterior, prior)
        self.assertAlmostEqual(sum(beliefs.players[seat].probs.values()), 1.0, places=8)

    def test_public_root_rebuilds_round_start_state(self) -> None:
        engine = NoLimitHoldemEngine(seed=404)
        engine.start_hand()
        _advance_to_flop(engine)

        seat_at_flop_start = int(engine.to_act)  # type: ignore[arg-type]
        engine.apply_action(Action(kind="check"))

        root = build_public_search_root(engine)
        self.assertEqual(root.street, "flop")
        self.assertEqual(root.round_start_engine.street, Street.FLOP)
        self.assertEqual(root.round_start_engine.to_act, seat_at_flop_start)
        self.assertEqual(len(root.round_start_engine.board), 3)
        self.assertTrue(all(action.street == "preflop" for action in root.round_start_engine.action_log))
        self.assertIn("street=flop", root.public_state_token)

        seat0 = root.beliefs.players[0]
        self.assertEqual(len(seat0.probs), 1176)
        self.assertAlmostEqual(sum(seat0.probs.values()), 1.0, places=8)


def _advance_to_flop(engine: NoLimitHoldemEngine) -> None:
    # Complete preflop by matching blinds and checking BB.
    engine.apply_action(Action(kind="call"))   # seat 3
    engine.apply_action(Action(kind="call"))   # seat 4
    engine.apply_action(Action(kind="call"))   # seat 5
    engine.apply_action(Action(kind="call"))   # seat 0
    engine.apply_action(Action(kind="call"))   # seat 1
    engine.apply_action(Action(kind="check"))  # seat 2

    if engine.street != Street.FLOP:
        raise AssertionError("expected flop after preflop completion")


def _pair_probability(distribution: dict[tuple[str, str], float]) -> float:
    return sum(prob for hand, prob in distribution.items() if hand[0][0] == hand[1][0])


if __name__ == "__main__":
    unittest.main()
