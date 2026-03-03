from dataclasses import dataclass, field
import random
from typing import Any, Mapping, Sequence

from pluribus_ri.core import Street
from pluribus_ri.solver import NLTHAbstractAction, NLTHAbstractGameFactory, NLTHAbstractGameState


@dataclass(frozen=True)
class BlueprintPolicy:
    """Playable policy derived from saved blueprint strategy snapshots."""

    iteration: int
    preflop_average: Mapping[str, Sequence[float]]
    postflop_current: Mapping[str, Sequence[float]]
    _preflop_distribution_cache: dict[tuple[str, int], tuple[float, ...] | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _postflop_distribution_cache: dict[tuple[str, int], tuple[float, ...] | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_snapshot_payload(cls, payload: dict[str, Any]) -> "BlueprintPolicy":
        iteration = int(payload["iteration"])
        preflop = _normalize_mapping(payload.get("preflop_average", {}))
        postflop = _normalize_mapping(payload.get("postflop_current", {}))
        return cls(
            iteration=iteration,
            preflop_average=preflop,
            postflop_current=postflop,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "preflop_average": self.preflop_average,
            "postflop_current": self.postflop_current,
            "preflop_infosets": len(self.preflop_average),
            "postflop_infosets": len(self.postflop_current),
        }

    def action_distribution(
        self,
        state: NLTHAbstractGameState,
    ) -> tuple[list[NLTHAbstractAction], Sequence[float]]:
        actions = list(state.legal_actions())
        if not actions:
            raise ValueError("state has no legal actions")

        seat = state.current_player()
        key = state.infoset_key(seat)

        if state.engine.street == Street.PREFLOP:
            distribution = self._lookup_distribution(
                table=self.preflop_average,
                cache=self._preflop_distribution_cache,
                key=key,
                expected_actions=len(actions),
            )
        else:
            distribution = self._lookup_distribution(
                table=self.postflop_current,
                cache=self._postflop_distribution_cache,
                key=key,
                expected_actions=len(actions),
            )

        if distribution is None:
            uniform = 1.0 / len(actions)
            distribution = tuple([uniform] * len(actions))

        return actions, distribution

    def select_action(
        self,
        state: NLTHAbstractGameState,
        rng: random.Random | None = None,
    ) -> NLTHAbstractAction:
        actions, distribution = self.action_distribution(state)
        if rng is None:
            best_idx = max(range(len(distribution)), key=lambda i: distribution[i])
            return actions[best_idx]
        sampled_idx = _sample_index(rng, distribution)
        return actions[sampled_idx]

    def _lookup_distribution(
        self,
        table: Mapping[str, Sequence[float]],
        cache: dict[tuple[str, int], tuple[float, ...] | None],
        key: str,
        expected_actions: int,
    ) -> tuple[float, ...] | None:
        cache_key = (key, expected_actions)
        if cache_key in cache:
            return cache[cache_key]

        probs = table.get(key)
        if probs is None or len(probs) != expected_actions:
            cache[cache_key] = None
            return None

        clipped: list[float] = []
        total = 0.0
        for value in probs:
            clipped_value = max(0.0, float(value))
            clipped.append(clipped_value)
            total += clipped_value
        if total <= 0.0:
            cache[cache_key] = None
            return None

        normalized = tuple(value / total for value in clipped)
        cache[cache_key] = normalized
        return normalized


def run_blueprint_self_play(
    policy: BlueprintPolicy,
    game_factory: NLTHAbstractGameFactory,
    num_hands: int,
    random_seed: int = 0,
) -> dict[str, Any]:
    if num_hands <= 0:
        raise ValueError("num_hands must be positive")

    rng = random.Random(random_seed)
    num_players = game_factory.game_config.num_players
    utility_sums = [0.0 for _ in range(num_players)]

    for _ in range(num_hands):
        state = game_factory.root_state()
        guard = 0
        while not state.is_terminal():
            guard += 1
            if guard > 500:
                raise RuntimeError("self-play hand exceeded action guard")
            action = policy.select_action(state, rng=rng)
            state = state.child(action)

        for player in range(num_players):
            utility_sums[player] += state.utility(player)

    return {
        "hands_played": num_hands,
        "utility_sums": utility_sums,
        "mean_utility_per_hand": [value / num_hands for value in utility_sums],
        "zero_sum_check": sum(utility_sums),
    }


def _sample_index(rng: random.Random, probs: Sequence[float]) -> int:
    draw = rng.random()
    running = 0.0
    for idx, prob in enumerate(probs):
        running += prob
        if draw <= running:
            return idx
    return len(probs) - 1


def _normalize_mapping(raw: object) -> dict[str, list[float] | tuple[float, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("snapshot strategy map must be a dictionary")

    normalized: dict[str, list[float] | tuple[float, ...]] = {}
    for key, values in raw.items():
        if not isinstance(key, str):
            raise ValueError("strategy map keys must be strings")
        if not isinstance(values, (list, tuple)):
            raise ValueError("strategy map values must be lists or tuples")
        normalized[key] = values
    return normalized
