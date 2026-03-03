from poker_game_runner.state import Observation
from poker_game_runner.utils import Range, HandType
import time
import random

BOT_NAME = "Python Bot" # Change this to your bot's name

class Bot:
  @classmethod
  def get_name_class(cls, path):
    return BOT_NAME

  def get_name(self):
      return BOT_NAME

  def act(self, obs: Observation):
    # Your code here
    return obs.get_max_raise() # All-in
