from dataclasses import dataclass, field
import random
import time

from pluribus_ri.blueprint import BlueprintPolicy
from pluribus_ri.core import Action, NoLimitHoldemEngine

from .nested_search import NestedUnsafeSearchConfig, NestedUnsafeSearcher
from .public_root import build_public_search_root


@dataclass(frozen=True)
class RuntimeSearchBenchmarkConfig:
    runs: int = 50
    seed_start: int = 0
    max_prefix_actions: int = 12
    random_action_seed: int = 0
    max_sampling_attempts: int = 1000
    search_config: NestedUnsafeSearchConfig = field(default_factory=NestedUnsafeSearchConfig)


@dataclass(frozen=True)
class RuntimeSearchBenchmarkResult:
    run_count: int
    attempted_states: int
    seed_start: int
    seeds_used: tuple[int, ...]
    max_prefix_actions: int
    latency_ms_min: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_max: float
    nodes_visited_mean: float
    nodes_visited_p95: float
    nodes_visited_max: int
    cfr_iterations_mean: float
    cfr_iterations_p95: float
    cfr_iterations_max: int
    stop_reason_counts: dict[str, int]
    street_counts: dict[str, int]
    chosen_action_kind_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_count": self.run_count,
            "attempted_states": self.attempted_states,
            "seed_start": self.seed_start,
            "seeds_used": list(self.seeds_used),
            "max_prefix_actions": self.max_prefix_actions,
            "latency_ms": {
                "min": round(self.latency_ms_min, 6),
                "mean": round(self.latency_ms_mean, 6),
                "p50": round(self.latency_ms_p50, 6),
                "p95": round(self.latency_ms_p95, 6),
                "p99": round(self.latency_ms_p99, 6),
                "max": round(self.latency_ms_max, 6),
            },
            "nodes_visited": {
                "mean": round(self.nodes_visited_mean, 6),
                "p95": round(self.nodes_visited_p95, 6),
                "max": self.nodes_visited_max,
            },
            "cfr_iterations": {
                "mean": round(self.cfr_iterations_mean, 6),
                "p95": round(self.cfr_iterations_p95, 6),
                "max": self.cfr_iterations_max,
            },
            "stop_reason_counts": dict(self.stop_reason_counts),
            "street_counts": dict(self.street_counts),
            "chosen_action_kind_counts": dict(self.chosen_action_kind_counts),
        }


def run_nested_search_benchmark(
    config: RuntimeSearchBenchmarkConfig,
    blueprint_policy: BlueprintPolicy | None = None,
) -> RuntimeSearchBenchmarkResult:
    if config.runs <= 0:
        raise ValueError("runs must be positive")
    if config.max_prefix_actions < 0:
        raise ValueError("max_prefix_actions must be >= 0")
    if config.max_sampling_attempts <= 0:
        raise ValueError("max_sampling_attempts must be positive")

    policy = blueprint_policy or BlueprintPolicy(iteration=0, preflop_average={}, postflop_current={})
    rng = random.Random(config.random_action_seed)

    latencies_ms: list[float] = []
    nodes_visited: list[int] = []
    cfr_iterations: list[int] = []
    stop_reason_counts: dict[str, int] = {}
    street_counts: dict[str, int] = {}
    chosen_action_kind_counts: dict[str, int] = {}
    seeds_used: list[int] = []

    attempts = 0
    candidate_seed = config.seed_start
    while len(latencies_ms) < config.runs and attempts < config.max_sampling_attempts:
        attempts += 1
        engine = NoLimitHoldemEngine(seed=candidate_seed)
        engine.start_hand()
        _play_random_prefix(
            engine=engine,
            rng=rng,
            max_actions=config.max_prefix_actions,
        )
        current_seed = candidate_seed
        candidate_seed += 1

        if engine.hand_complete or engine.to_act is None:
            continue

        root = build_public_search_root(engine)
        searcher = NestedUnsafeSearcher(config.search_config)

        started = time.perf_counter()
        result = searcher.search(
            root=root,
            blueprint_policy=policy,
            acting_seat=int(engine.to_act),  # type: ignore[arg-type]
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        latencies_ms.append(elapsed_ms)
        nodes_visited.append(int(result.nodes_visited))
        cfr_iterations.append(int(result.cfr_iterations))
        seeds_used.append(current_seed)

        stop_reason_counts[result.stopping_reason] = stop_reason_counts.get(result.stopping_reason, 0) + 1
        street_key = engine.street.value
        street_counts[street_key] = street_counts.get(street_key, 0) + 1
        action_key = result.chosen_action.kind
        chosen_action_kind_counts[action_key] = chosen_action_kind_counts.get(action_key, 0) + 1

    if len(latencies_ms) < config.runs:
        raise RuntimeError(
            "failed to collect requested number of benchmark states "
            f"({len(latencies_ms)} / {config.runs})"
        )

    return RuntimeSearchBenchmarkResult(
        run_count=config.runs,
        attempted_states=attempts,
        seed_start=config.seed_start,
        seeds_used=tuple(seeds_used),
        max_prefix_actions=config.max_prefix_actions,
        latency_ms_min=min(latencies_ms),
        latency_ms_mean=_mean(latencies_ms),
        latency_ms_p50=_percentile(latencies_ms, 0.50),
        latency_ms_p95=_percentile(latencies_ms, 0.95),
        latency_ms_p99=_percentile(latencies_ms, 0.99),
        latency_ms_max=max(latencies_ms),
        nodes_visited_mean=_mean([float(value) for value in nodes_visited]),
        nodes_visited_p95=_percentile([float(value) for value in nodes_visited], 0.95),
        nodes_visited_max=max(nodes_visited),
        cfr_iterations_mean=_mean([float(value) for value in cfr_iterations]),
        cfr_iterations_p95=_percentile([float(value) for value in cfr_iterations], 0.95),
        cfr_iterations_max=max(cfr_iterations),
        stop_reason_counts=stop_reason_counts,
        street_counts=street_counts,
        chosen_action_kind_counts=chosen_action_kind_counts,
    )


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    if percentile < 0.0 or percentile > 1.0:
        raise ValueError("percentile must be in [0, 1]")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = percentile * (len(ordered) - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(ordered) - 1)
    fraction = rank - lower_idx
    return ordered[lower_idx] * (1.0 - fraction) + ordered[upper_idx] * fraction


def _play_random_prefix(
    engine: NoLimitHoldemEngine,
    rng: random.Random,
    max_actions: int,
) -> None:
    prefix_len = rng.randint(0, max_actions)
    for _ in range(prefix_len):
        if engine.hand_complete:
            return
        engine.apply_action(_sample_random_legal_action(engine, rng))


def _sample_random_legal_action(engine: NoLimitHoldemEngine, rng: random.Random) -> Action:
    legal = engine.get_legal_actions()
    options: list[Action] = []

    if legal.can_fold:
        options.append(Action(kind="fold"))

    if legal.can_check:
        options.append(Action(kind="check"))
    elif legal.call_amount > 0:
        options.append(Action(kind="call"))

    if legal.min_raise_to is not None and legal.max_raise_to is not None:
        min_raise = int(legal.min_raise_to)
        max_raise = int(legal.max_raise_to)
        candidates = {min_raise, max_raise, (min_raise + max_raise) // 2}
        if max_raise > min_raise:
            candidates.add(rng.randint(min_raise, max_raise))
        for amount in sorted(candidates):
            options.append(Action(kind="raise", amount=amount))

    if not options:
        raise RuntimeError("no legal action options available for benchmark state sampler")
    return rng.choice(options)
