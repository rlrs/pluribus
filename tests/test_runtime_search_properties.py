import copy
import random
import unittest

from pluribus_ri.abstraction import NLTHAbstractGameBuilder
from pluribus_ri.core import Action, NoLimitHoldemEngine
from pluribus_ri.runtime_search import (
    NestedUnsafeSearchConfig,
    NestedUnsafeSearcher,
    PseudoHarmonicRaiseTranslator,
    SubgameStoppingRules,
    build_public_search_root,
    insert_off_tree_action,
)
from pluribus_ri.solver import NLTHAbstractGameState


class RuntimeSearchPropertyTests(unittest.TestCase):
    def test_round_prefix_maps_replay_consistent_under_random_histories(self) -> None:
        rng = random.Random(20260303)
        searcher = NestedUnsafeSearcher(
            NestedUnsafeSearchConfig(
                random_seed=3,
                stopping_rules=SubgameStoppingRules(
                    min_cfr_iterations=1,
                    max_cfr_iterations=1,
                    max_nodes_touched=50_000,
                    max_wallclock_ms=1000,
                    leaf_max_depth=8,
                ),
            )
        )

        checked = 0
        seed = 500
        while checked < 30:
            engine = NoLimitHoldemEngine(seed=seed)
            engine.start_hand()
            _play_random_prefix(engine=engine, rng=rng, max_actions=22)
            seed += 1

            if engine.hand_complete or engine.to_act is None:
                continue

            root = build_public_search_root(engine)
            prefix = searcher._build_current_round_prefix_maps(root)  # type: ignore[attr-defined]

            self.assertEqual(prefix.off_tree_insertions, len(prefix.inserted_actions))

            replay = copy.deepcopy(root.round_start_engine)
            street_actions = [
                action
                for action in root.current_engine.action_log
                if action.street == root.street
            ]
            for observed in street_actions:
                token = searcher.builder.public_state_token(replay)
                self.assertIn(token, prefix.forced_actions)

                forced = prefix.forced_actions[token]
                _assert_action_applies(replay, forced)

                inserted = prefix.inserted_actions.get(token, ())
                if inserted:
                    raise_amounts = [int(action.amount) for action in inserted if action.kind == "raise"]
                    self.assertEqual(raise_amounts, sorted(raise_amounts))
                    self.assertEqual(len(raise_amounts), len(set(raise_amounts)))
                    if observed.kind == "raise":
                        self.assertIn(int(observed.amount), raise_amounts)

                replay.apply_action(Action(kind=observed.kind, amount=int(observed.amount)))

            self.assertEqual(
                searcher.builder.public_state_token(replay),
                searcher.builder.public_state_token(root.current_engine),
            )
            checked += 1

        self.assertEqual(checked, 30)

    def test_off_tree_translation_and_insertion_invariants_under_random_raises(self) -> None:
        rng = random.Random(20260304)
        builder = NLTHAbstractGameBuilder()
        translator = PseudoHarmonicRaiseTranslator()

        checked = 0
        seed = 1000
        while checked < 80:
            engine = NoLimitHoldemEngine(seed=seed)
            engine.start_hand()
            _play_random_prefix(engine=engine, rng=rng, max_actions=16)
            seed += 1

            if engine.hand_complete:
                continue

            legal = engine.get_legal_actions()
            if legal.min_raise_to is None or legal.max_raise_to is None:
                continue

            state = NLTHAbstractGameState(
                engine=engine,
                root_stacks=tuple(engine._hand_starting_stacks),
                abstraction_builder=builder,
            )
            abstract = [action.to_engine_action() for action in state.legal_actions()]
            raise_actions = sorted(
                [action for action in abstract if action.kind == "raise"],
                key=lambda action: int(action.amount),
            )
            if len(raise_actions) < 2:
                continue

            observed_amount = rng.randint(int(legal.min_raise_to), int(legal.max_raise_to))
            observed = Action(kind="raise", amount=observed_amount)
            translation = translator.translate(engine, observed, abstract)

            self.assertTrue(
                any(
                    action.kind == translation.translated_action.kind
                    and int(action.amount) == int(translation.translated_action.amount)
                    for action in abstract
                )
            )
            self.assertEqual(
                int(translation.distance),
                abs(int(translation.translated_action.amount) - observed_amount),
            )
            self.assertGreater(len(translation.mixed_strategy), 0)
            self.assertAlmostEqual(
                sum(weight for _, weight in translation.mixed_strategy),
                1.0,
                places=6,
            )

            exact_present = any(int(action.amount) == observed_amount for action in raise_actions)
            if exact_present:
                self.assertFalse(translation.was_off_tree)
                self.assertEqual(int(translation.distance), 0)
            else:
                self.assertTrue(translation.was_off_tree)

            insertion = insert_off_tree_action(engine, observed, abstract)
            if exact_present:
                self.assertFalse(insertion.was_inserted)
            else:
                self.assertTrue(insertion.was_inserted)
                inserted_raises = [int(action.amount) for action in insertion.actions if action.kind == "raise"]
                self.assertIn(observed_amount, inserted_raises)
                self.assertEqual(inserted_raises, sorted(inserted_raises))

            checked += 1

        self.assertEqual(checked, 80)


def _play_random_prefix(
    engine: NoLimitHoldemEngine,
    rng: random.Random,
    max_actions: int,
) -> None:
    prefix_len = rng.randint(0, max_actions)
    for _ in range(prefix_len):
        if engine.hand_complete:
            return
        engine.apply_action(_sample_random_legal_action(engine, rng))


def _sample_random_legal_action(engine: NoLimitHoldemEngine, rng: random.Random) -> Action:
    legal = engine.get_legal_actions()
    options: list[Action] = []

    if legal.can_fold:
        options.append(Action(kind="fold"))

    if legal.can_check:
        options.append(Action(kind="check"))
    elif legal.call_amount > 0:
        options.append(Action(kind="call"))

    if legal.min_raise_to is not None and legal.max_raise_to is not None:
        min_raise = int(legal.min_raise_to)
        max_raise = int(legal.max_raise_to)
        candidates = {min_raise, max_raise, (min_raise + max_raise) // 2}
        if max_raise > min_raise:
            candidates.add(rng.randint(min_raise, max_raise))
        for amount in sorted(candidates):
            options.append(Action(kind="raise", amount=amount))

    if not options:
        raise AssertionError("no legal options sampled from legal action descriptor")
    return rng.choice(options)


def _assert_action_applies(engine: NoLimitHoldemEngine, action: Action) -> None:
    candidate = copy.deepcopy(engine)
    try:
        candidate.apply_action(Action(kind=action.kind, amount=int(action.amount)))
    except ValueError as exc:
        raise AssertionError(f"expected action to be legal at replay state: {action}") from exc


if __name__ == "__main__":
    unittest.main()
