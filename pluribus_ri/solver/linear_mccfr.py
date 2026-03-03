from dataclasses import dataclass
import random
from typing import Callable, Generic, Protocol, Sequence, TypeVar

from .regret_table import LazyIntRegretTable


ActionT = TypeVar("ActionT")


class ExtensiveGameState(Protocol, Generic[ActionT]):
    """Minimal interface required by the external-sampling MCCFR trainer."""

    def is_terminal(self) -> bool:
        ...

    def utility(self, player: int) -> float:
        ...

    def is_chance_node(self) -> bool:
        ...

    def chance_outcomes(self) -> Sequence[tuple[ActionT, float]]:
        ...

    def current_player(self) -> int:
        ...

    def legal_actions(self) -> Sequence[ActionT]:
        ...

    def child(self, action: ActionT) -> "ExtensiveGameState[ActionT]":
        ...

    def infoset_key(self, player: int) -> str:
        ...


@dataclass
class MCCFRConfig:
    iterations: int
    random_seed: int | None = 0
    linear_weighting: bool = True

    # Optional global decay for long runs.
    discount_interval: int = 0
    regret_discount_factor: float = 1.0
    average_strategy_discount_factor: float = 1.0

    # Negative-regret pruning scaffolding.
    prune_after_iteration: int = 0
    negative_regret_pruning_threshold: int = -300_000_000
    explore_all_actions_probability: float = 0.05

    track_average_strategy: bool = True


@dataclass
class TrainingStats:
    iterations_completed: int = 0
    traversals_completed: int = 0
    nodes_touched: int = 0


class ExternalSamplingLinearMCCFR:
    """
    External-sampling MCCFR with linear iteration weighting.

    This is Phase-1 scaffolding: the traversal/update logic is implemented,
    while full Pluribus abstraction/search integration is added later.
    """

    def __init__(self, regret_table: LazyIntRegretTable, config: MCCFRConfig) -> None:
        if config.iterations <= 0:
            raise ValueError("iterations must be positive")
        if config.discount_interval < 0:
            raise ValueError("discount_interval must be >= 0")
        if config.explore_all_actions_probability < 0 or config.explore_all_actions_probability > 1:
            raise ValueError("explore_all_actions_probability must be in [0, 1]")

        self.regret_table = regret_table
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.stats = TrainingStats()

    def train(
        self,
        root_state_factory: Callable[[], ExtensiveGameState[ActionT]],
        num_players: int,
    ) -> TrainingStats:
        return self.train_steps(
            root_state_factory=root_state_factory,
            num_players=num_players,
            iterations=self.config.iterations,
        )

    def train_steps(
        self,
        root_state_factory: Callable[[], ExtensiveGameState[ActionT]],
        num_players: int,
        iterations: int,
        on_iteration_end: Callable[[int, TrainingStats], None] | None = None,
    ) -> TrainingStats:
        if num_players <= 0:
            raise ValueError("num_players must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        start_iteration = self.stats.iterations_completed
        end_iteration = start_iteration + iterations

        for iteration in range(start_iteration + 1, end_iteration + 1):
            for traverser in range(num_players):
                root_state = root_state_factory()
                self._traverse(
                    state=root_state,
                    traverser=traverser,
                    iteration=iteration,
                )
                self.stats.traversals_completed += 1

            if self.config.discount_interval and iteration % self.config.discount_interval == 0:
                self.regret_table.scale_all_regrets(self.config.regret_discount_factor)
                if self.config.track_average_strategy:
                    self.regret_table.scale_all_average_strategies(
                        self.config.average_strategy_discount_factor
                    )

            self.stats.iterations_completed = iteration
            if on_iteration_end is not None:
                on_iteration_end(iteration, self.stats)

        return self.stats

    def _traverse(
        self,
        state: ExtensiveGameState[ActionT],
        traverser: int,
        iteration: int,
    ) -> float:
        self.stats.nodes_touched += 1

        if state.is_terminal():
            return state.utility(traverser)

        if state.is_chance_node():
            action = self._sample_chance_action(state.chance_outcomes())
            return self._traverse(state.child(action), traverser=traverser, iteration=iteration)

        player = state.current_player()
        actions = list(state.legal_actions())
        if not actions:
            raise RuntimeError("non-terminal state has no legal actions")

        key = state.infoset_key(player)
        num_actions = len(actions)
        regrets = self.regret_table.get_regrets(key, num_actions)
        strategy = self.regret_table.current_strategy(key, num_actions)

        iteration_weight = float(iteration) if self.config.linear_weighting else 1.0

        if player == traverser:
            candidate_indices = self._candidate_action_indices(regrets, iteration)
            traverser_strategy = _renormalize_subset(strategy, candidate_indices)

            action_values = [0.0] * num_actions
            node_value = 0.0

            for idx in candidate_indices:
                child_value = self._traverse(
                    state.child(actions[idx]),
                    traverser=traverser,
                    iteration=iteration,
                )
                action_values[idx] = child_value
                node_value += traverser_strategy[idx] * child_value

            for idx in candidate_indices:
                regret_delta = (action_values[idx] - node_value) * iteration_weight
                self.regret_table.add_regret(
                    key=key,
                    action_index=idx,
                    delta=regret_delta,
                    num_actions=num_actions,
                )

            if self.config.track_average_strategy:
                self.regret_table.accumulate_average_strategy(
                    key=key,
                    strategy=traverser_strategy,
                    num_actions=num_actions,
                    weight=iteration_weight,
                )

            return node_value

        if self.config.track_average_strategy:
            self.regret_table.accumulate_average_strategy(
                key=key,
                strategy=strategy,
                num_actions=num_actions,
                weight=iteration_weight,
            )

        sampled_index = _sample_from_distribution(self.rng, strategy)
        return self._traverse(
            state.child(actions[sampled_index]),
            traverser=traverser,
            iteration=iteration,
        )

    def _sample_chance_action(self, outcomes: Sequence[tuple[ActionT, float]]) -> ActionT:
        if not outcomes:
            raise RuntimeError("chance node has no outcomes")
        probabilities = [max(0.0, prob) for _, prob in outcomes]
        total = sum(probabilities)
        if total <= 0.0:
            # Fallback to uniform when probabilities are not normalized.
            probabilities = [1.0 for _ in outcomes]
            total = float(len(outcomes))
        normalized = [prob / total for prob in probabilities]
        index = _sample_from_distribution(self.rng, normalized)
        return outcomes[index][0]

    def _candidate_action_indices(self, regrets: list[int], iteration: int) -> list[int]:
        indices = list(range(len(regrets)))
        if self.config.prune_after_iteration <= 0 or iteration < self.config.prune_after_iteration:
            return indices

        if self.rng.random() < self.config.explore_all_actions_probability:
            return indices

        kept = [
            idx
            for idx, regret in enumerate(regrets)
            if regret > self.config.negative_regret_pruning_threshold
        ]
        return kept if kept else indices


def _sample_from_distribution(rng: random.Random, probs: Sequence[float]) -> int:
    draw = rng.random()
    running = 0.0
    for idx, prob in enumerate(probs):
        running += prob
        if draw <= running:
            return idx
    return len(probs) - 1


def _renormalize_subset(strategy: Sequence[float], kept_indices: Sequence[int]) -> list[float]:
    out = [0.0] * len(strategy)
    subtotal = sum(strategy[idx] for idx in kept_indices)
    if subtotal <= 0.0:
        uniform = 1.0 / len(kept_indices)
        for idx in kept_indices:
            out[idx] = uniform
        return out

    for idx in kept_indices:
        out[idx] = strategy[idx] / subtotal
    return out
