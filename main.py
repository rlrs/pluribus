import sys
from example_bots.python import random_bot, never_bluff_bot
from test import run_benchmark, run_table
import my_bot

bots = [random_bot, never_bluff_bot, my_bot]

mode = sys.argv[1] if len(sys.argv) > 1 else "table"

if mode == "benchmark":
    run_count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_benchmark(bots, run_count)
else:
    run_table(bots)
