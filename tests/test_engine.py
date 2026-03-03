import unittest

from pluribus_ri.core import Action, NoLimitHoldemEngine, Street


class NoLimitHoldemEngineTests(unittest.TestCase):
    def test_clone_for_simulation_is_state_equivalent_and_independent_for_actions(self) -> None:
        engine = NoLimitHoldemEngine(seed=21)
        engine.start_hand()
        engine.apply_action(Action(kind="call"))
        engine.apply_action(Action(kind="call"))

        clone = engine.clone_for_simulation()

        self.assertEqual(clone.street, engine.street)
        self.assertEqual(clone.to_act, engine.to_act)
        self.assertEqual(clone.board, engine.board)
        self.assertEqual(clone.stacks, engine.stacks)
        self.assertEqual(len(clone.action_log), len(engine.action_log))

        original_to_act = engine.to_act
        original_action_log_len = len(engine.action_log)
        original_player_stacks = [player.stack for player in engine.players]

        legal = clone.get_legal_actions()
        if legal.min_raise_to is not None:
            action = Action(kind="raise", amount=int(legal.min_raise_to))
        elif legal.can_check:
            action = Action(kind="check")
        elif legal.call_amount > 0:
            action = Action(kind="call")
        else:
            action = Action(kind="fold")
        clone.apply_action(action)

        self.assertNotEqual(len(clone.action_log), len(engine.action_log))
        self.assertEqual(len(engine.action_log), original_action_log_len)
        self.assertEqual([player.stack for player in engine.players], original_player_stacks)
        self.assertEqual(engine.to_act, original_to_act)

    def test_preflop_legal_actions_for_utg(self) -> None:
        engine = NoLimitHoldemEngine(seed=7)
        engine.start_hand()

        self.assertEqual(engine.street, Street.PREFLOP)
        self.assertEqual(engine.to_act, 3)

        legal = engine.get_legal_actions()
        self.assertTrue(legal.can_fold)
        self.assertFalse(legal.can_check)
        self.assertEqual(legal.call_amount, 100)
        self.assertEqual(legal.min_raise_to, 200)
        self.assertEqual(legal.max_raise_to, 10_000)

    def test_hand_history_replay_is_deterministic(self) -> None:
        engine = NoLimitHoldemEngine(seed=13)
        engine.start_hand()

        # Preflop folds to big blind.
        for _ in range(5):
            engine.apply_action(Action(kind="fold"))

        self.assertTrue(engine.hand_complete)

        history = engine.export_hand_history()
        replay = NoLimitHoldemEngine.replay_hand(history)

        self.assertTrue(replay.hand_complete)
        self.assertEqual(replay.stacks, engine.stacks)
        self.assertEqual(replay.winners, engine.winners)
        self.assertEqual(replay.board, engine.board)
        self.assertEqual(len(replay.action_log), len(engine.action_log))

    def test_all_in_showdown_and_side_pot_distribution(self) -> None:
        engine = NoLimitHoldemEngine(seed=1)
        deck_prefix = [
            "2c", "Kc", "3c", "4c", "5c", "Ah",
            "2d", "Kd", "3d", "4d", "5d", "Ad",
            "2h", "7c", "8d", "9s", "Jc",
        ]
        engine.start_hand(deck_cards=deck_prefix)

        # Seats to act preflop: 3,4,5,0,1,2
        engine.apply_action(Action(kind="fold"))      # seat 3
        engine.apply_action(Action(kind="fold"))      # seat 4
        engine.apply_action(Action(kind="fold"))      # seat 5
        engine.apply_action(Action(kind="raise", amount=10_000))  # seat 0 all-in
        engine.apply_action(Action(kind="fold"))      # seat 1 folds small blind
        engine.apply_action(Action(kind="call"))      # seat 2 calls all-in

        self.assertTrue(engine.hand_complete)
        self.assertEqual(len(engine.board), 5)

        self.assertEqual(engine.stacks[0], 20_050)
        self.assertEqual(engine.stacks[1], 9_950)
        self.assertEqual(engine.stacks[2], 0)
        self.assertEqual(sum(engine.stacks), 60_000)
        self.assertEqual(engine.winners, [0])


if __name__ == "__main__":
    unittest.main()
