import copy
import unittest

from pluribus_ri.blueprint import BlueprintPolicy
from pluribus_ri.core import Action, NoLimitHoldemEngine, Street
from pluribus_ri.runtime_search import (
    FrozenOwnActionMap,
    LeafContinuationConfig,
    NearestRaiseActionTranslator,
    NestedUnsafeSearchConfig,
    NestedUnsafeSearcher,
    PseudoHarmonicRaiseTranslator,
    SubgameStoppingRules,
    build_public_search_root,
    insert_off_tree_action,
)


class NestedUnsafeSearchTests(unittest.TestCase):
    def test_search_returns_legal_action_with_stopping_metadata(self) -> None:
        engine = NoLimitHoldemEngine(seed=515)
        engine.start_hand()
        root = build_public_search_root(engine)

        policy = BlueprintPolicy(iteration=0, preflop_average={}, postflop_current={})
        searcher = NestedUnsafeSearcher(
            NestedUnsafeSearchConfig(
                random_seed=9,
                stopping_rules=SubgameStoppingRules(
                    min_cfr_iterations=1,
                    max_cfr_iterations=2,
                    max_nodes_touched=5000,
                    max_wallclock_ms=1000,
                    leaf_max_depth=6,
                ),
                leaf_continuation=LeafContinuationConfig(
                    rollout_count=1,
                    max_actions_per_rollout=96,
                    random_seed=11,
                ),
            )
        )
        result = searcher.search(root=root, blueprint_policy=policy, acting_seat=int(engine.to_act))  # type: ignore[arg-type]

        self.assertGreaterEqual(len(result.action_values), 2)
        self.assertEqual(result.cfr_iterations, 2)
        self.assertEqual(result.stopping_reason, "max_iterations")
        self.assertGreater(result.nodes_visited, 0)
        self.assertGreaterEqual(result.rollouts_run, 1)
        self.assertEqual(len(result.root_strategy), len(result.action_values))

        replay = copy.deepcopy(engine)
        replay.apply_action(result.chosen_action)
        self.assertNotEqual(replay.total_pot, 0)

    def test_frozen_own_action_map_tracks_current_round_actions(self) -> None:
        engine = NoLimitHoldemEngine(seed=616)
        engine.start_hand()
        _advance_to_flop(engine)
        self.assertEqual(engine.to_act, 1)

        engine.apply_action(Action(kind="check"))  # seat 1
        legal = engine.get_legal_actions()  # seat 2
        self.assertIsNotNone(legal.min_raise_to)
        engine.apply_action(Action(kind="raise", amount=int(legal.min_raise_to)))  # seat 2
        engine.apply_action(Action(kind="fold"))  # seat 3
        engine.apply_action(Action(kind="fold"))  # seat 4
        engine.apply_action(Action(kind="fold"))  # seat 5
        engine.apply_action(Action(kind="fold"))  # seat 0
        self.assertEqual(engine.to_act, 1)

        root = build_public_search_root(engine)
        frozen = FrozenOwnActionMap.from_public_root(root=root, seat=1)
        self.assertEqual(frozen.count, 1)
        self.assertTrue(any(action.kind == "check" for action in frozen.actions_by_public_token.values()))

        policy = BlueprintPolicy(iteration=0, preflop_average={}, postflop_current={})
        searcher = NestedUnsafeSearcher(
            NestedUnsafeSearchConfig(
                random_seed=4,
                freeze_own_actions=True,
                stopping_rules=SubgameStoppingRules(
                    min_cfr_iterations=1,
                    max_cfr_iterations=1,
                    max_nodes_touched=5000,
                    max_wallclock_ms=1000,
                    leaf_max_depth=6,
                ),
                leaf_continuation=LeafContinuationConfig(
                    rollout_count=1,
                    max_actions_per_rollout=96,
                    random_seed=21,
                ),
            )
        )
        result = searcher.search(root=root, blueprint_policy=policy, acting_seat=1)
        self.assertEqual(result.frozen_own_action_count, 1)

    def test_action_translators_and_insertion(self) -> None:
        engine = NoLimitHoldemEngine(seed=7)
        engine.start_hand()

        legal = engine.get_legal_actions()
        self.assertIsNotNone(legal.min_raise_to)
        observed = Action(kind="raise", amount=int(legal.min_raise_to) + 37)  # type: ignore[arg-type]
        # clamp to legal range
        observed = Action(
            kind="raise",
            amount=min(int(legal.max_raise_to), max(int(legal.min_raise_to), observed.amount)),  # type: ignore[arg-type]
        )

        abstract = [
            Action(kind="call"),
            Action(kind="raise", amount=int(legal.min_raise_to)),  # type: ignore[arg-type]
            Action(kind="raise", amount=int(legal.max_raise_to)),  # type: ignore[arg-type]
        ]

        nearest = NearestRaiseActionTranslator().translate(engine, observed, abstract)
        self.assertIn(nearest.translated_action, abstract[1:])

        pseudo = PseudoHarmonicRaiseTranslator().translate(engine, observed, abstract)
        self.assertTrue(pseudo.was_off_tree or pseudo.distance == 0)
        self.assertGreater(len(pseudo.mixed_strategy), 0)
        self.assertAlmostEqual(sum(weight for _, weight in pseudo.mixed_strategy), 1.0, places=6)

        inserted = insert_off_tree_action(engine, observed, abstract)
        self.assertTrue(inserted.was_inserted or any(a.amount == observed.amount for a in abstract))
        if inserted.was_inserted:
            self.assertTrue(any(a.kind == "raise" and a.amount == observed.amount for a in inserted.actions))


def _advance_to_flop(engine: NoLimitHoldemEngine) -> None:
    engine.apply_action(Action(kind="call"))   # seat 3
    engine.apply_action(Action(kind="call"))   # seat 4
    engine.apply_action(Action(kind="call"))   # seat 5
    engine.apply_action(Action(kind="call"))   # seat 0
    engine.apply_action(Action(kind="call"))   # seat 1
    engine.apply_action(Action(kind="check"))  # seat 2

    if engine.street != Street.FLOP:
        raise AssertionError("expected flop")


if __name__ == "__main__":
    unittest.main()
