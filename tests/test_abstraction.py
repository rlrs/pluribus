import unittest

from pluribus_ri.abstraction import PublicStateKey, encode_infoset_key, normalize_action_history


class AbstractionTests(unittest.TestCase):
    def test_public_state_key_is_deterministic(self) -> None:
        key = PublicStateKey(
            street="flop",
            board_cards=("Ah", "Kd", "2c"),
            to_act=4,
            pot=750,
            current_bet=200,
            stacks=(10000, 9800, 9600, 9400, 9200, 9000),
            contributed_street=(0, 0, 0, 200, 200, 0),
            active_mask=(1, 1, 1, 1, 1, 0),
            action_history=("p3:raise:200", "p4:call:200"),
        )

        token = key.to_token()
        self.assertIn("street=flop", token)
        self.assertIn("board=AhKd2c", token)

        infoset = encode_infoset_key(seat=4, private_bucket=123, public_state=key)
        self.assertTrue(infoset.startswith("p4|b123|street=flop"))

    def test_normalize_action_history(self) -> None:
        normalized = normalize_action_history(["  P1:Raise:300 ", "", "p2:CALL:300  "])
        self.assertEqual(normalized, ("p1:raise:300", "p2:call:300"))


if __name__ == "__main__":
    unittest.main()
