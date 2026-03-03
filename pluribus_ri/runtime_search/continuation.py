from dataclasses import dataclass, field
import random
from typing import Literal, Sequence

from pluribus_ri.abstraction import ActionAbstractionConfig, HistoryScope, NLTHAbstractGameBuilder
from pluribus_ri.blueprint import BlueprintPolicy
from pluribus_ri.core import Action, NoLimitHoldemEngine
from pluribus_ri.solver import NLTHAbstractAction, NLTHAbstractGameState


ContinuationStrategy = Literal["blueprint", "fold_biased", "call_biased", "raise_biased"]

DEFAULT_CONTINUATION_MIX: tuple[tuple[ContinuationStrategy, float], ...] = (
    ("blueprint", 0.25),
    ("fold_biased", 0.25),
    ("call_biased", 0.25),
    ("raise_biased", 0.25),
)


@dataclass(frozen=True)
class LeafContinuationConfig:
    rollout_count: int = 8
    max_actions_per_rollout: int = 128
    random_seed: int = 0
    strategy_mix: tuple[tuple[ContinuationStrategy, float], ...] = DEFAULT_CONTINUATION_MIX
    history_scope: HistoryScope = "street"
    abstraction_config: ActionAbstractionConfig = field(default_factory=ActionAbstractionConfig)


class ContinuationLeafEvaluator:
    """
    Leaf evaluator that mixes continuation strategies for each player.

    This implements the Phase-3 continuation-policy interface used when subgame
    search is depth-limited and cannot resolve to terminal utility directly.
    """

    def __init__(
        self,
        blueprint_policy: BlueprintPolicy,
        config: LeafContinuationConfig | None = None,
    ) -> None:
        self.policy = blueprint_policy
        self.config = config or LeafContinuationConfig()
        if self.config.rollout_count <= 0:
            raise ValueError("rollout_count must be positive")
        if self.config.max_actions_per_rollout <= 0:
            raise ValueError("max_actions_per_rollout must be positive")
        self._mix = _normalized_mix(self.config.strategy_mix)
        self._rng = random.Random(self.config.random_seed)
        self.builder = NLTHAbstractGameBuilder(
            abstraction_config=self.config.abstraction_config,
            history_scope=self.config.history_scope,
        )
        self.rollouts_run = 0

    def evaluate(
        self,
        engine: NoLimitHoldemEngine,
        player: int,
        root_stacks: tuple[int, ...],
    ) -> float:
        total = 0.0
        for _ in range(self.config.rollout_count):
            sim = engine.clone_for_simulation()
            strategies = [
                _sample_strategy(self._rng, self._mix)
                for _ in range(sim.num_players)
            ]
            total += self._single_rollout(
                engine=sim,
                player=player,
                root_stacks=root_stacks,
                seat_strategies=strategies,
            )
            self.rollouts_run += 1
        return total / self.config.rollout_count

    def _single_rollout(
        self,
        engine: NoLimitHoldemEngine,
        player: int,
        root_stacks: tuple[int, ...],
        seat_strategies: Sequence[ContinuationStrategy],
    ) -> float:
        for _ in range(self.config.max_actions_per_rollout):
            if engine.hand_complete:
                return float(engine.stacks[player] - root_stacks[player])

            seat = engine.to_act
            if seat is None:
                raise RuntimeError("rollout reached invalid non-terminal state")
            action = _sample_continuation_action(
                engine=engine,
                seat_strategy=seat_strategies[seat],
                policy=self.policy,
                builder=self.builder,
                rng=self._rng,
            )
            engine.apply_action(action)

        # Truncated rollout fallback: use current chip delta as proxy.
        return float(engine.stacks[player] - root_stacks[player])


def _sample_continuation_action(
    engine: NoLimitHoldemEngine,
    seat_strategy: ContinuationStrategy,
    policy: BlueprintPolicy,
    builder: NLTHAbstractGameBuilder,
    rng: random.Random,
) -> Action:
    state = NLTHAbstractGameState(
        engine=engine,
        root_stacks=tuple(engine._hand_starting_stacks),
        abstraction_builder=builder,
    )
    actions, base_probs = policy.action_distribution(state)
    biased = apply_continuation_bias(actions, base_probs, seat_strategy)
    index = _sample_index(rng, biased)
    return actions[index].to_engine_action()


def apply_continuation_bias(
    actions: Sequence[NLTHAbstractAction],
    base_probs: Sequence[float],
    strategy: ContinuationStrategy,
) -> list[float]:
    if len(actions) != len(base_probs):
        raise ValueError("actions/probability length mismatch")
    if not actions:
        raise ValueError("actions must be non-empty")

    if strategy == "blueprint":
        return _normalize_nonnegative(base_probs)

    weights: list[float] = []
    for action, base in zip(actions, base_probs):
        value = max(0.0, float(base))
        if strategy == "fold_biased" and action.kind == "fold":
            value *= 3.0
        elif strategy == "call_biased" and action.kind in {"check", "call"}:
            value *= 3.0
        elif strategy == "raise_biased" and action.kind == "raise":
            value *= 3.0
        weights.append(value)

    return _normalize_nonnegative(weights)


def _normalize_nonnegative(values: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0.0:
        uniform = 1.0 / len(clipped)
        return [uniform for _ in clipped]
    return [value / total for value in clipped]


def _normalized_mix(
    mix: Sequence[tuple[ContinuationStrategy, float]],
) -> tuple[tuple[ContinuationStrategy, float], ...]:
    if not mix:
        raise ValueError("strategy mix must be non-empty")
    total = sum(max(0.0, float(weight)) for _, weight in mix)
    if total <= 0.0:
        uniform = 1.0 / len(mix)
        return tuple((name, uniform) for name, _ in mix)
    return tuple((name, max(0.0, float(weight)) / total) for name, weight in mix)


def _sample_strategy(
    rng: random.Random,
    mix: Sequence[tuple[ContinuationStrategy, float]],
) -> ContinuationStrategy:
    draw = rng.random()
    running = 0.0
    for strategy, weight in mix:
        running += weight
        if draw <= running:
            return strategy
    return mix[-1][0]


def _sample_index(rng: random.Random, probs: Sequence[float]) -> int:
    draw = rng.random()
    running = 0.0
    for idx, prob in enumerate(probs):
        running += prob
        if draw <= running:
            return idx
    return len(probs) - 1
