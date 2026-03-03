from typing import Sequence


def current_strategy_from_regret_array(regrets: Sequence[int]) -> list[float]:
    num_actions = len(regrets)
    positive = [max(0.0, float(v)) for v in regrets]
    normalizer = sum(positive)
    if normalizer <= 0.0:
        uniform = 1.0 / num_actions
        return [uniform] * num_actions
    return [value / normalizer for value in positive]


def accumulate_strategy_sums_inplace(
    sums: list[float],
    strategy: Sequence[float],
    weight: float,
) -> None:
    if len(strategy) != len(sums):
        raise ValueError("strategy length mismatch")
    for i, prob in enumerate(strategy):
        sums[i] += weight * float(prob)
