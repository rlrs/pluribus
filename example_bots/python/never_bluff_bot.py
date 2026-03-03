from poker_game_runner.state import Observation
from poker_game_runner.utils import Range, HandType
"""
  This bot will only raise, if its hand is better than the cards on the table.
"""

BOT_NAME = "Never Bluff Bot"

class Bot:
    @classmethod
    def get_name_class(cls, path):
        return BOT_NAME

    def get_name(self):
        return BOT_NAME

    def act(self, obs: Observation):

        my_hand = obs.get_my_hand_type()
        board = obs.get_board_hand_type()

        if my_hand > board:
            return obs.get_min_raise()
        else:
            return 1
