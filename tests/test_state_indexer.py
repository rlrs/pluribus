import unittest

from pluribus_ri.abstraction import (
    build_public_state_key,
    encode_engine_infoset_key,
    private_hand_bucket,
    private_hand_bucket_with_policy,
)
from pluribus_ri.core import Action, NoLimitHoldemEngine, Street


class StateIndexerTests(unittest.TestCase):
    def test_round_scoped_history_and_infoset_encoding(self) -> None:
        engine = NoLimitHoldemEngine(seed=17)
        engine.start_hand()

        preflop_bucket = private_hand_bucket(engine, seat=3)
        self.assertGreaterEqual(preflop_bucket, 0)

        # Move from preflop to flop without ending the hand.
        engine.apply_action(Action(kind="call"))   # seat 3
        engine.apply_action(Action(kind="call"))   # seat 4
        engine.apply_action(Action(kind="call"))   # seat 5
        engine.apply_action(Action(kind="call"))   # seat 0
        engine.apply_action(Action(kind="call"))   # seat 1
        engine.apply_action(Action(kind="check"))  # seat 2

        self.assertEqual(engine.street, Street.FLOP)

        street_key = build_public_state_key(engine, history_scope="street")
        self.assertEqual(street_key.street, "flop")
        self.assertEqual(street_key.action_history, ())

        all_key = build_public_state_key(engine, history_scope="all")
        self.assertGreater(len(all_key.action_history), 0)
        self.assertTrue(all_key.action_history[0].startswith("preflop:p"))

        # Add one flop action and ensure round-scoped history updates.
        engine.apply_action(Action(kind="check"))  # seat 1
        flop_key = build_public_state_key(engine, history_scope="street")
        self.assertEqual(flop_key.action_history, ("p1:check:0",))

        infoset = encode_engine_infoset_key(engine=engine, seat=2, history_scope="street")
        self.assertTrue(infoset.startswith("p2|b"))
        self.assertIn("street=flop", infoset)

        postflop_bucket = private_hand_bucket(engine, seat=2)
        self.assertGreaterEqual(postflop_bucket, 0)

    def test_canonical169_preflop_bucket_policy_covers_all_hand_classes(self) -> None:
        engine = NoLimitHoldemEngine(seed=42)
        engine.start_hand()

        deck = NoLimitHoldemEngine.full_deck()
        buckets: set[int] = set()
        for i in range(len(deck)):
            for j in range(i + 1, len(deck)):
                engine.players[0].hole_cards = (deck[i], deck[j])
                bucket = private_hand_bucket_with_policy(
                    engine=engine,
                    seat=0,
                    preflop_bucket_policy="canonical169",
                )
                buckets.add(bucket)

        self.assertEqual(len(buckets), 169)
        self.assertEqual(min(buckets), 0)
        self.assertEqual(max(buckets), 168)

    def test_texture_postflop_bucket_policy_is_deterministic(self) -> None:
        engine = NoLimitHoldemEngine(seed=99)
        engine.start_hand()

        engine.players[0].hole_cards = ("Ah", "Qh")
        engine.board = ["Th", "9h", "2c"]

        legacy = private_hand_bucket_with_policy(
            engine=engine,
            seat=0,
            postflop_bucket_policy="legacy",
        )
        texture_a = private_hand_bucket_with_policy(
            engine=engine,
            seat=0,
            postflop_bucket_policy="texture_v1",
        )
        texture_b = private_hand_bucket_with_policy(
            engine=engine,
            seat=0,
            postflop_bucket_policy="texture_v1",
        )

        self.assertEqual(texture_a, texture_b)
        self.assertNotEqual(texture_a, legacy)
        self.assertGreaterEqual(texture_a, 0)


if __name__ == "__main__":
    unittest.main()
