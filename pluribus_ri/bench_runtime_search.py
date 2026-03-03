import argparse
import json
from pathlib import Path
from typing import Sequence

from pluribus_ri.runtime_search import (
    LeafContinuationConfig,
    NestedUnsafeSearchConfig,
    RuntimeSearchBenchmarkConfig,
    SubgameStoppingRules,
    run_nested_search_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark runtime search latency and resolver stopping behavior."
    )

    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--max-prefix-actions", type=int, default=12)
    parser.add_argument("--random-action-seed", type=int, default=0)
    parser.add_argument("--max-sampling-attempts", type=int, default=2000)

    parser.add_argument("--search-random-seed", type=int, default=0)
    parser.add_argument("--history-scope", choices=["street", "all"], default="street")
    parser.add_argument("--freeze-own-actions", action="store_true", default=True)
    parser.add_argument("--no-freeze-own-actions", action="store_false", dest="freeze_own_actions")
    parser.add_argument("--insert-off-tree-actions", action="store_true", default=True)
    parser.add_argument("--no-insert-off-tree-actions", action="store_false", dest="insert_off_tree_actions")

    parser.add_argument("--min-cfr-iterations", type=int, default=8)
    parser.add_argument("--max-cfr-iterations", type=int, default=64)
    parser.add_argument("--max-nodes-touched", type=int, default=75_000)
    parser.add_argument("--max-wallclock-ms", type=int, default=250)
    parser.add_argument("--leaf-max-depth", type=int, default=12)

    parser.add_argument("--rollout-count", type=int, default=8)
    parser.add_argument("--max-actions-per-rollout", type=int, default=128)
    parser.add_argument("--leaf-random-seed", type=int, default=0)

    parser.add_argument("--linear-weighting", action="store_true", default=True)
    parser.add_argument("--no-linear-weighting", action="store_false", dest="linear_weighting")

    parser.add_argument("--output-path", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    search_config = NestedUnsafeSearchConfig(
        random_seed=args.search_random_seed,
        freeze_own_actions=args.freeze_own_actions,
        insert_off_tree_actions=args.insert_off_tree_actions,
        history_scope=args.history_scope,
        stopping_rules=SubgameStoppingRules(
            min_cfr_iterations=args.min_cfr_iterations,
            max_cfr_iterations=args.max_cfr_iterations,
            max_nodes_touched=args.max_nodes_touched,
            max_wallclock_ms=args.max_wallclock_ms,
            leaf_max_depth=args.leaf_max_depth,
        ),
        leaf_continuation=LeafContinuationConfig(
            rollout_count=args.rollout_count,
            max_actions_per_rollout=args.max_actions_per_rollout,
            random_seed=args.leaf_random_seed,
        ),
        linear_weighting=args.linear_weighting,
    )
    benchmark_config = RuntimeSearchBenchmarkConfig(
        runs=args.runs,
        seed_start=args.seed_start,
        max_prefix_actions=args.max_prefix_actions,
        random_action_seed=args.random_action_seed,
        max_sampling_attempts=args.max_sampling_attempts,
        search_config=search_config,
    )

    result = run_nested_search_benchmark(benchmark_config)
    payload = result.to_dict()

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
