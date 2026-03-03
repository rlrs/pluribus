from poker_game_runner.state import Observation
from poker_game_runner.utils import Range, HandType

BOT_NAME = "Chicken Bot"

class Bot:
    @classmethod
    def get_name_class(cls, path):
        return BOT_NAME

    def get_name(self):
        return BOT_NAME

    def act(self, obs: Observation):
        # This bot checks if the amount to call is greater than 0
        # that is, it checks if anyone has raised.
        # If that is the case, it chickens out and folds!

        amount_to_call = obs.get_call_size()
        if amount_to_call > 0:
            return 0
        else:
            return 1
