import unittest

from pluribus_ri.blueprint import BlueprintPolicy, run_blueprint_self_play
from pluribus_ri.solver import NLTHAbstractGameFactory, NLTHActionAbstractionConfig, NLTHGameConfig


class BlueprintPolicyTests(unittest.TestCase):
    def test_policy_uses_snapshot_distribution_when_infoset_present(self) -> None:
        factory = NLTHAbstractGameFactory(
            game_config=NLTHGameConfig(random_seed=4),
            abstraction_config=NLTHActionAbstractionConfig(max_raise_actions=2),
        )
        state = factory.root_state()

        actions = list(state.legal_actions())
        self.assertGreater(len(actions), 1)

        key = state.infoset_key(state.current_player())
        probs = [0.0 for _ in actions]
        probs[-1] = 1.0

        policy = BlueprintPolicy.from_snapshot_payload(
            {
                "iteration": 7,
                "preflop_average": {key: probs},
                "postflop_current": {},
            }
        )

        selected = policy.select_action(state)
        self.assertEqual(selected, actions[-1])

    def test_policy_falls_back_to_uniform_when_infoset_missing(self) -> None:
        factory = NLTHAbstractGameFactory(
            game_config=NLTHGameConfig(random_seed=6),
            abstraction_config=NLTHActionAbstractionConfig(max_raise_actions=2),
        )
        state = factory.root_state()

        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})
        actions, distribution = policy.action_distribution(state)

        self.assertEqual(len(actions), len(distribution))
        self.assertAlmostEqual(sum(distribution), 1.0, places=6)
        for value in distribution:
            self.assertAlmostEqual(value, 1.0 / len(distribution), places=6)

    def test_blueprint_self_play_runs_end_to_end_and_zero_sum(self) -> None:
        factory = NLTHAbstractGameFactory(game_config=NLTHGameConfig(random_seed=10))
        policy = BlueprintPolicy(iteration=1, preflop_average={}, postflop_current={})

        result = run_blueprint_self_play(
            policy=policy,
            game_factory=factory,
            num_hands=4,
            random_seed=99,
        )

        self.assertEqual(result["hands_played"], 4)
        self.assertEqual(len(result["utility_sums"]), 6)
        self.assertAlmostEqual(float(result["zero_sum_check"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
