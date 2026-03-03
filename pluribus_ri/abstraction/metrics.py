from dataclasses import asdict, dataclass
import math
import random
from typing import Sequence

import eval7

from pluribus_ri.core import NoLimitHoldemEngine

from .state_indexer import PostflopBucketPolicy, PreflopBucketPolicy, private_hand_bucket_with_policy


@dataclass(frozen=True)
class PostflopBucketCalibrationReport:
    policy: PostflopBucketPolicy
    samples: int
    seed: int
    board_lengths: tuple[int, ...]
    unique_buckets: int
    bucket_entropy: float
    mean_bucket_size: float
    max_bucket_size: int
    mean_bucket_score_std: float
    weighted_bucket_score_std: float
    score_bucket_spearman: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bucket_entropy"] = round(self.bucket_entropy, 6)
        payload["mean_bucket_size"] = round(self.mean_bucket_size, 6)
        payload["mean_bucket_score_std"] = round(self.mean_bucket_score_std, 6)
        payload["weighted_bucket_score_std"] = round(self.weighted_bucket_score_std, 6)
        payload["score_bucket_spearman"] = round(self.score_bucket_spearman, 6)
        return payload


def build_postflop_bucket_calibration_report(
    policy: PostflopBucketPolicy,
    samples: int = 3000,
    seed: int = 0,
    board_lengths: Sequence[int] = (3, 4, 5),
    preflop_bucket_policy: PreflopBucketPolicy = "legacy",
) -> PostflopBucketCalibrationReport:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not board_lengths:
        raise ValueError("board_lengths cannot be empty")
    if any(length not in {3, 4, 5} for length in board_lengths):
        raise ValueError("board_lengths must contain only postflop sizes 3/4/5")

    rng = random.Random(seed)
    deck = NoLimitHoldemEngine.full_deck()
    engine = NoLimitHoldemEngine(seed=seed)
    engine.start_hand()

    per_bucket_scores: dict[int, list[int]] = {}
    bucket_series: list[int] = []
    score_series: list[int] = []

    for _ in range(samples):
        board_len = rng.choice(tuple(board_lengths))
        dealt = rng.sample(deck, 2 + board_len)
        hole = (dealt[0], dealt[1])
        board = tuple(dealt[2:])

        engine.players[0].hole_cards = hole
        engine.board = list(board)

        bucket = private_hand_bucket_with_policy(
            engine=engine,
            seat=0,
            preflop_bucket_policy=preflop_bucket_policy,
            postflop_bucket_policy=policy,
        )
        score = int(eval7.evaluate([eval7.Card(card) for card in (*hole, *board)]))

        per_bucket_scores.setdefault(bucket, []).append(score)
        bucket_series.append(bucket)
        score_series.append(score)

    bucket_sizes = [len(values) for values in per_bucket_scores.values()]
    unique_buckets = len(bucket_sizes)
    probabilities = [size / samples for size in bucket_sizes]
    entropy = -sum(prob * math.log2(prob) for prob in probabilities if prob > 0.0)

    bucket_stds = [_population_std(values) for values in per_bucket_scores.values()]
    weighted_std = sum(std * len(values) for std, values in zip(bucket_stds, per_bucket_scores.values())) / samples
    mean_std = sum(bucket_stds) / len(bucket_stds)
    spearman = _spearman_rank_correlation(bucket_series, score_series)

    return PostflopBucketCalibrationReport(
        policy=policy,
        samples=samples,
        seed=seed,
        board_lengths=tuple(board_lengths),
        unique_buckets=unique_buckets,
        bucket_entropy=entropy,
        mean_bucket_size=samples / unique_buckets,
        max_bucket_size=max(bucket_sizes),
        mean_bucket_score_std=mean_std,
        weighted_bucket_score_std=weighted_std,
        score_bucket_spearman=spearman,
    )


def compare_postflop_bucket_policies(
    policies: Sequence[PostflopBucketPolicy] = ("legacy", "texture_v1"),
    samples: int = 3000,
    seed: int = 0,
    board_lengths: Sequence[int] = (3, 4, 5),
    preflop_bucket_policy: PreflopBucketPolicy = "legacy",
) -> dict[str, PostflopBucketCalibrationReport]:
    reports: dict[str, PostflopBucketCalibrationReport] = {}
    for policy in policies:
        reports[str(policy)] = build_postflop_bucket_calibration_report(
            policy=policy,
            samples=samples,
            seed=seed,
            board_lengths=board_lengths,
            preflop_bucket_policy=preflop_bucket_policy,
        )
    return reports


def _population_std(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _spearman_rank_correlation(xs: Sequence[int], ys: Sequence[int]) -> float:
    if len(xs) != len(ys):
        raise ValueError("input lengths must match")
    if len(xs) < 2:
        return 0.0

    x_ranks = _average_ranks(xs)
    y_ranks = _average_ranks(ys)
    return _pearson(x_ranks, y_ranks)


def _average_ranks(values: Sequence[int]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)

    pos = 0
    while pos < len(values):
        end = pos + 1
        while end < len(values) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + end - 1) / 2.0 + 1.0
        for idx in order[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)
