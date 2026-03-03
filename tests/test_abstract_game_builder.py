import re
import unittest

from pluribus_ri.abstraction import (
    ActionAbstractionConfig,
    NLTHAbstractGameBuilder,
    private_hand_bucket_with_policy,
)
from pluribus_ri.core import Action, NoLimitHoldemEngine, Street


class AbstractGameBuilderTests(unittest.TestCase):
    def test_preflop_action_specs_obey_config_and_are_deduplicated(self) -> None:
        engine = NoLimitHoldemEngine(seed=11)
        engine.start_hand()

        builder = NLTHAbstractGameBuilder(
            abstraction_config=ActionAbstractionConfig(
                preflop_raise_multipliers=(2.0, 3.0, 5.0, 10.0),
                max_raise_actions=3,
            ),
            history_scope="street",
        )

        actions = builder.legal_action_specs(engine)
        kinds = [kind for kind, _ in actions]
        raise_targets = [amount for kind, amount in actions if kind == "raise"]

        self.assertIn("fold", kinds)
        self.assertIn("call", kinds)
        self.assertGreaterEqual(len(raise_targets), 1)
        self.assertLessEqual(len(raise_targets), 3)
        self.assertEqual(len(raise_targets), len(set(raise_targets)))

    def test_infoset_and_public_token_follow_history_scope(self) -> None:
        engine = NoLimitHoldemEngine(seed=12)
        engine.start_hand()

        # Move from preflop to flop.
        engine.apply_action(Action(kind="call"))   # seat 3
        engine.apply_action(Action(kind="call"))   # seat 4
        engine.apply_action(Action(kind="call"))   # seat 5
        engine.apply_action(Action(kind="call"))   # seat 0
        engine.apply_action(Action(kind="call"))   # seat 1
        engine.apply_action(Action(kind="check"))  # seat 2
        self.assertEqual(engine.street, Street.FLOP)

        street_builder = NLTHAbstractGameBuilder(history_scope="street")
        all_builder = NLTHAbstractGameBuilder(history_scope="all")

        seat = int(engine.to_act)  # type: ignore[arg-type]
        street_infoset = street_builder.infoset_key(engine=engine, seat=seat)
        all_infoset = all_builder.infoset_key(engine=engine, seat=seat)

        self.assertIn("street=flop", street_infoset)
        self.assertIn("|hist=", street_infoset)
        self.assertIn("street=flop", all_infoset)

        street_token = street_builder.public_state_token(engine)
        all_token = all_builder.public_state_token(engine)
        self.assertIn("hist=", street_token)
        self.assertIn("hist=preflop:", all_token)

    def test_infoset_respects_preflop_bucket_policy(self) -> None:
        engine = NoLimitHoldemEngine(seed=31)
        engine.start_hand()
        seat = int(engine.to_act)  # type: ignore[arg-type]

        legacy_builder = NLTHAbstractGameBuilder(
            abstraction_config=ActionAbstractionConfig(preflop_bucket_policy="legacy"),
            history_scope="street",
        )
        canonical_builder = NLTHAbstractGameBuilder(
            abstraction_config=ActionAbstractionConfig(preflop_bucket_policy="canonical169"),
            history_scope="street",
        )

        legacy_infoset = legacy_builder.infoset_key(engine=engine, seat=seat)
        canonical_infoset = canonical_builder.infoset_key(engine=engine, seat=seat)

        legacy_bucket = _extract_bucket(legacy_infoset)
        canonical_bucket = _extract_bucket(canonical_infoset)

        self.assertEqual(
            legacy_bucket,
            private_hand_bucket_with_policy(engine=engine, seat=seat, preflop_bucket_policy="legacy"),
        )
        self.assertEqual(
            canonical_bucket,
            private_hand_bucket_with_policy(engine=engine, seat=seat, preflop_bucket_policy="canonical169"),
        )
        self.assertNotEqual(legacy_infoset, canonical_infoset)

    def test_flop_specific_raise_fractions_override_generic_postflop_table(self) -> None:
        engine = NoLimitHoldemEngine(seed=12)
        engine.start_hand()

        engine.apply_action(Action(kind="call"))   # seat 3
        engine.apply_action(Action(kind="call"))   # seat 4
        engine.apply_action(Action(kind="call"))   # seat 5
        engine.apply_action(Action(kind="call"))   # seat 0
        engine.apply_action(Action(kind="call"))   # seat 1
        engine.apply_action(Action(kind="check"))  # seat 2
        self.assertEqual(engine.street, Street.FLOP)

        generic_only = NLTHAbstractGameBuilder(
            abstraction_config=ActionAbstractionConfig(
                postflop_pot_raise_fractions=(2.0,),
                include_all_in=False,
                max_raise_actions=4,
            )
        )
        with_flop_override = NLTHAbstractGameBuilder(
            abstraction_config=ActionAbstractionConfig(
                postflop_pot_raise_fractions=(2.0,),
                flop_pot_raise_fractions=(0.1,),
                include_all_in=False,
                max_raise_actions=4,
            )
        )

        generic_raises = [
            amount
            for kind, amount in generic_only.legal_action_specs(engine)
            if kind == "raise"
        ]
        override_raises = [
            amount
            for kind, amount in with_flop_override.legal_action_specs(engine)
            if kind == "raise"
        ]

        self.assertTrue(any(amount >= 1200 for amount in generic_raises))
        self.assertTrue(all(amount <= 200 for amount in override_raises))
        self.assertNotEqual(generic_raises, override_raises)


def _extract_bucket(infoset: str) -> int:
    match = re.search(r"\|b(-?\d+)\|", infoset)
    if match is None:
        raise AssertionError(f"infoset does not contain bucket token: {infoset}")
    return int(match.group(1))


if __name__ == "__main__":
    unittest.main()
