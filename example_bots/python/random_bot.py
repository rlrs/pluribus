from poker_game_runner.state import Observation
from poker_game_runner.utils import Range, HandType
import random

"""
This bot makes a random number which is either 0 or 1. If it is 1, it will always raise the minimum amount that it possibly can. Otherwise, it attempts to call.
"""

BOT_NAME = "Random Bot"

class Bot:
  @classmethod
  def get_name_class(cls, path):
    return BOT_NAME

  def get_name(self):
      return BOT_NAME

  def act(self, obs: Observation):
    shouldRaise = random.randint(0,1)
    if (shouldRaise == 1):
      return obs.get_min_raise()
    else:
      return 1