from poker_game_runner.state import Observation
from poker_game_runner.utils import Range, HandType
"""
  This bot will always raise the pot with it's entire cash stack (going "all-in")
"""

BOT_NAME = "Bluff Bot"

class Bot:
    @classmethod
    def get_name_class(cls, path):
      return BOT_NAME

    def get_name(self):
        return BOT_NAME

    def act(self, obs: Observation):
        return obs.get_max_raise()
