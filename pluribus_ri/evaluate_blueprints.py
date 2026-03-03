import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from pluribus_ri.abstraction import load_abstraction_tables_config
from pluribus_ri.blueprint import (
    AIVATConfig,
    BaselineStrategy,
    ExploitabilityProxyConfig,
    LeagueEvaluationConfig,
    SUPPORTED_BASELINES,
    load_blueprint_policy,
    run_exploitability_proxy_report,
    run_one_vs_field_league,
)
from pluribus_ri.solver import NLTHActionAbstractionConfig, NLTHGameConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 5 blueprint evaluation: league and exploitability-proxy reports."
    )
    parser.add_argument(
        "--report",
        choices=["league", "exploitability_proxy", "all"],
        default="league",
        help="Evaluation report type to produce.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help="Entrant as NAME=PATH_TO_BLUEPRINT_JSON. Provide at least one (two for league/all reports).",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--num-hands-per-seat", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument("--max-actions-per-hand", type=int, default=500)
    parser.add_argument(
        "--aivat",
        action="store_true",
        help="Enable AIVAT-style action-correction adjusted CI reporting.",
    )
    parser.add_argument("--aivat-rollouts-per-action", type=int, default=1)
    parser.add_argument("--aivat-max-actions-per-rollout", type=int, default=64)
    parser.add_argument("--aivat-max-branching", type=int, default=8)
    parser.add_argument(
        "--aivat-own-decisions-only",
        action="store_true",
        help="Apply AIVAT corrections only at candidate-seat decisions.",
    )
    parser.add_argument(
        "--control-variate-baseline",
        choices=[*SUPPORTED_BASELINES, "none"],
        default="none",
        help="Enable control-variate adjusted CI reporting with the selected baseline (or none).",
    )
    parser.add_argument(
        "--proxy-baselines",
        default=",".join(SUPPORTED_BASELINES),
        help="Comma-separated exploitability proxy baseline pool.",
    )
    parser.add_argument(
        "--proxy-control-variate-baseline",
        choices=[*SUPPORTED_BASELINES, "none"],
        default=None,
        help="Optional proxy-specific control variate baseline override.",
    )
    parser.add_argument(
        "--proxy-num-hands-per-seat",
        type=int,
        default=None,
        help="Override hands per seat for exploitability proxy report only.",
    )
    parser.add_argument(
        "--proxy-random-seed",
        type=int,
        default=None,
        help="Override random seed for exploitability proxy report only.",
    )

    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--starting-stack", type=int, default=10_000)
    parser.add_argument("--button", type=int, default=0)
    parser.add_argument("--history-scope", choices=["street", "all"], default="street")

    parser.add_argument(
        "--abstraction-config",
        default=None,
        help="Path to abstraction table JSON. Overrides abstraction CLI flags.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    min_entries = 1 if args.report == "exploitability_proxy" else 2
    policy_paths = _parse_policy_specs(args.policy, min_entries=min_entries)
    policies = {name: load_blueprint_policy(path) for name, path in policy_paths.items()}

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

    game_config = NLTHGameConfig(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.starting_stack,
        button=args.button,
        random_seed=args.random_seed,
        history_scope=history_scope,  # type: ignore[arg-type]
    )
    abstraction_config = NLTHActionAbstractionConfig(
        preflop_raise_multipliers=preflop_raise_multipliers,
        postflop_pot_raise_fractions=postflop_pot_raise_fractions,
        flop_pot_raise_fractions=flop_pot_raise_fractions,
        turn_pot_raise_fractions=turn_pot_raise_fractions,
        river_pot_raise_fractions=river_pot_raise_fractions,
        include_all_in=include_all_in,
        max_raise_actions=max_raise_actions,
        preflop_bucket_policy=preflop_bucket_policy,  # type: ignore[arg-type]
        postflop_bucket_policy=postflop_bucket_policy,  # type: ignore[arg-type]
    )
    eval_config = LeagueEvaluationConfig(
        num_hands_per_seat=args.num_hands_per_seat,
        random_seed=args.random_seed,
        confidence_z=args.confidence_z,
        sample_actions=not args.deterministic_actions,
        max_actions_per_hand=args.max_actions_per_hand,
        control_variate_baseline=_parse_optional_baseline(args.control_variate_baseline),
        aivat_config=(
            AIVATConfig(
                rollout_count_per_action=args.aivat_rollouts_per_action,
                max_actions_per_rollout=args.aivat_max_actions_per_rollout,
                max_branching_for_correction=args.aivat_max_branching,
                include_opponent_decisions=not args.aivat_own_decisions_only,
            )
            if args.aivat
            else None
        ),
    )

    proxy_control_variate_baseline = _parse_optional_baseline(
        args.proxy_control_variate_baseline
        if args.proxy_control_variate_baseline is not None
        else args.control_variate_baseline
    )
    proxy_config = ExploitabilityProxyConfig(
        num_hands_per_seat=args.proxy_num_hands_per_seat or args.num_hands_per_seat,
        random_seed=args.proxy_random_seed if args.proxy_random_seed is not None else args.random_seed,
        confidence_z=args.confidence_z,
        sample_actions=not args.deterministic_actions,
        max_actions_per_hand=args.max_actions_per_hand,
        baseline_policies=cast(
            tuple[str, ...],
            _parse_proxy_baselines(args.proxy_baselines),
        ),
        control_variate_baseline=proxy_control_variate_baseline,
        aivat_config=eval_config.aivat_config,
    )

    entrants = {name: str(path) for name, path in policy_paths.items()}
    if args.report == "league":
        result = run_one_vs_field_league(
            policies=policies,
            config=eval_config,
            game_config=game_config,
            abstraction_config=abstraction_config,
        )
        result["entrants"] = entrants
    elif args.report == "exploitability_proxy":
        result = run_exploitability_proxy_report(
            policies=policies,
            config=proxy_config,
            game_config=game_config,
            abstraction_config=abstraction_config,
        )
        result["entrants"] = entrants
    else:
        league = run_one_vs_field_league(
            policies=policies,
            config=eval_config,
            game_config=game_config,
            abstraction_config=abstraction_config,
        )
        proxy = run_exploitability_proxy_report(
            policies=policies,
            config=proxy_config,
            game_config=game_config,
            abstraction_config=abstraction_config,
        )
        result = {
            "format": "combined_phase5",
            "entrants": entrants,
            "league": league,
            "exploitability_proxy": proxy,
        }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parse_policy_specs(raw_specs: Sequence[str], *, min_entries: int) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in raw_specs:
        spec = raw.strip()
        if not spec:
            continue
        name, sep, path_text = spec.partition("=")
        if sep == "":
            raise ValueError("invalid --policy value; expected NAME=PATH")
        policy_name = name.strip()
        if not policy_name:
            raise ValueError("invalid --policy value; policy name is empty")
        if policy_name in parsed:
            raise ValueError(f"duplicate policy name: {policy_name}")
        policy_path = Path(path_text.strip())
        if not policy_path.exists():
            raise ValueError(f"blueprint path does not exist: {policy_path}")
        parsed[policy_name] = policy_path
    if len(parsed) < min_entries:
        if min_entries == 1:
            raise ValueError("provide at least one --policy NAME=PATH entry")
        raise ValueError(f"provide at least {min_entries} --policy NAME=PATH entries")
    return parsed


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


def _parse_proxy_baselines(raw: str) -> tuple[str, ...]:
    baselines = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not baselines:
        raise ValueError("proxy baseline pool must not be empty")

    unique: list[str] = []
    for baseline in baselines:
        if baseline in unique:
            continue
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(
                f"unsupported proxy baseline '{baseline}', supported: {', '.join(SUPPORTED_BASELINES)}"
            )
        unique.append(baseline)
    return tuple(unique)


def _parse_optional_baseline(raw: str | None) -> BaselineStrategy | None:
    if raw is None or raw == "none":
        return None
    if raw not in SUPPORTED_BASELINES:
        raise ValueError(f"unsupported baseline strategy: {raw}")
    return cast(BaselineStrategy, raw)


if __name__ == "__main__":
    raise SystemExit(main())
