import argparse
import json
from typing import Sequence

from pluribus_ri.abstraction import load_abstraction_tables_config
from pluribus_ri.training import (
    Phase1RunConfig,
    Phase2RunConfig,
    run_phase1_training,
    run_phase2_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run blueprint training (Phase 2 default) with external-sampling Linear MCCFR."
    )

    parser.add_argument("--phase", choices=["1", "2"], default="2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--snapshot-interval", type=int, default=25)
    parser.add_argument("--self-play-hands", type=int, default=24)
    parser.add_argument("--self-play-seed", type=int, default=0)

    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--history-scope", choices=["street", "all"], default="street")
    parser.add_argument(
        "--abstraction-config",
        default=None,
        help="Path to JSON abstraction table config (overrides history/action abstraction CLI flags).",
    )

    parser.add_argument("--preflop-raise-multipliers", default="2,3,5,10")
    parser.add_argument("--postflop-pot-fractions", default="0.5,1,2")
    parser.add_argument("--flop-pot-fractions", default=None)
    parser.add_argument("--turn-pot-fractions", default=None)
    parser.add_argument("--river-pot-fractions", default=None)
    parser.add_argument("--max-raise-actions", type=int, default=4)
    parser.add_argument("--no-all-in", action="store_true")
    parser.add_argument("--preflop-bucket-policy", choices=["legacy", "canonical169"], default="legacy")
    parser.add_argument("--postflop-bucket-policy", choices=["legacy", "texture_v1"], default="legacy")

    parser.add_argument("--linear-weighting", action="store_true", default=True)
    parser.add_argument("--no-linear-weighting", action="store_false", dest="linear_weighting")
    parser.add_argument("--discount-interval", type=int, default=0)
    parser.add_argument("--regret-discount-factor", type=float, default=1.0)
    parser.add_argument("--average-strategy-discount-factor", type=float, default=1.0)
    parser.add_argument("--prune-after-iteration", type=int, default=0)
    parser.add_argument("--negative-regret-pruning-threshold", type=int, default=-300000000)
    parser.add_argument("--explore-all-actions-probability", type=float, default=0.05)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir or ("artifacts/phase2" if args.phase == "2" else "artifacts/phase1")
    history_scope = args.history_scope
    preflop_raise_multipliers = _parse_float_tuple(args.preflop_raise_multipliers)
    postflop_pot_raise_fractions = _parse_float_tuple(args.postflop_pot_fractions)
    flop_pot_raise_fractions = _parse_optional_float_tuple(args.flop_pot_fractions)
    turn_pot_raise_fractions = _parse_optional_float_tuple(args.turn_pot_fractions)
    river_pot_raise_fractions = _parse_optional_float_tuple(args.river_pot_fractions)
    include_all_in = not args.no_all_in
    max_raise_actions = args.max_raise_actions
    preflop_bucket_policy = args.preflop_bucket_policy
    postflop_bucket_policy = args.postflop_bucket_policy

    if args.abstraction_config:
        tables = load_abstraction_tables_config(args.abstraction_config)
        history_scope = tables.history_scope
        preflop_raise_multipliers = tables.action.preflop_raise_multipliers
        postflop_pot_raise_fractions = tables.action.postflop_pot_raise_fractions
        flop_pot_raise_fractions = tables.action.flop_pot_raise_fractions
        turn_pot_raise_fractions = tables.action.turn_pot_raise_fractions
        river_pot_raise_fractions = tables.action.river_pot_raise_fractions
        include_all_in = tables.action.include_all_in
        max_raise_actions = tables.action.max_raise_actions
        preflop_bucket_policy = tables.action.preflop_bucket_policy
        postflop_bucket_policy = tables.action.postflop_bucket_policy

    common_kwargs = dict(
        output_dir=output_dir,
        iterations=args.iterations,
        checkpoint_interval=args.checkpoint_interval,
        snapshot_interval=args.snapshot_interval,
        random_seed=args.random_seed,
        history_scope=history_scope,
        preflop_raise_multipliers=preflop_raise_multipliers,
        postflop_pot_raise_fractions=postflop_pot_raise_fractions,
        flop_pot_raise_fractions=flop_pot_raise_fractions,
        turn_pot_raise_fractions=turn_pot_raise_fractions,
        river_pot_raise_fractions=river_pot_raise_fractions,
        include_all_in=include_all_in,
        max_raise_actions=max_raise_actions,
        preflop_bucket_policy=preflop_bucket_policy,
        postflop_bucket_policy=postflop_bucket_policy,
        linear_weighting=args.linear_weighting,
        discount_interval=args.discount_interval,
        regret_discount_factor=args.regret_discount_factor,
        average_strategy_discount_factor=args.average_strategy_discount_factor,
        prune_after_iteration=args.prune_after_iteration,
        negative_regret_pruning_threshold=args.negative_regret_pruning_threshold,
        explore_all_actions_probability=args.explore_all_actions_probability,
    )

    if args.phase == "1":
        summary = run_phase1_training(Phase1RunConfig(**common_kwargs))
    else:
        summary = run_phase2_training(
            Phase2RunConfig(
                **common_kwargs,
                self_play_hands=args.self_play_hands,
                self_play_seed=args.self_play_seed,
            )
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("at least one numeric value is required")
    return tuple(values)


def _parse_optional_float_tuple(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    return _parse_float_tuple(stripped)


if __name__ == "__main__":
    raise SystemExit(main())
