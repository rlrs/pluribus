from dataclasses import dataclass
import unittest

from pluribus_ri.core import engine_kernels
from pluribus_ri.core import _py_engine_kernels


@dataclass
class _DummyPlayer:
    stack: int
    hole_cards: tuple[str, str] | None
    folded: bool = False
    all_in: bool = False
    contributed_total: int = 0


class EngineKernelParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.players = [
            _DummyPlayer(stack=100, hole_cards=("As", "Kd"), contributed_total=25),
            _DummyPlayer(stack=0, hole_cards=("2c", "2d"), all_in=True, contributed_total=100),
            _DummyPlayer(stack=50, hole_cards=("Qh", "Qs"), contributed_total=60),
            _DummyPlayer(stack=20, hole_cards=None, folded=True, contributed_total=0),
            _DummyPlayer(stack=10, hole_cards=("9c", "9d"), folded=True, contributed_total=15),
            _DummyPlayer(stack=120, hole_cards=("Tc", "Th"), contributed_total=10),
        ]
        self.num_players = len(self.players)

    def test_live_player_count_and_last_matches_reference(self) -> None:
        got = engine_kernels.live_player_count_and_last(self.players, self.num_players)
        expected = _py_engine_kernels.live_player_count_and_last(self.players, self.num_players)
        self.assertEqual(got, expected)

    def test_all_live_players_all_in_matches_reference(self) -> None:
        got = engine_kernels.all_live_players_all_in(self.players, self.num_players)
        expected = _py_engine_kernels.all_live_players_all_in(self.players, self.num_players)
        self.assertEqual(got, expected)

    def test_eligible_and_pending_filters_match_reference(self) -> None:
        pending = {0, 1, 2, 3, 7}

        got_eligible = engine_kernels.eligible_seats(self.players, self.num_players, exclude_seat=2)
        exp_eligible = _py_engine_kernels.eligible_seats(self.players, self.num_players, exclude_seat=2)
        self.assertEqual(got_eligible, exp_eligible)

        got_filtered = engine_kernels.filter_pending_eligible(pending, self.players, self.num_players)
        exp_filtered = _py_engine_kernels.filter_pending_eligible(pending, self.players, self.num_players)
        self.assertEqual(got_filtered, exp_filtered)

    def test_next_pending_from_matches_reference(self) -> None:
        pending = {0, 2, 5}
        start = 0
        got = engine_kernels.next_pending_from(start, pending, self.players, self.num_players)
        expected = _py_engine_kernels.next_pending_from(start, pending, self.players, self.num_players)
        self.assertEqual(got, expected)

    def test_loader_exports_flag(self) -> None:
        self.assertIsInstance(engine_kernels.USING_CYTHON_ENGINE_KERNELS, bool)

    def test_total_pot_matches_reference(self) -> None:
        got = engine_kernels.total_pot(self.players, self.num_players)
        expected = _py_engine_kernels.total_pot(self.players, self.num_players)
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
