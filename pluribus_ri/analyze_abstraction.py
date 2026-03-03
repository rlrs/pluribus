import argparse
import json
from pathlib import Path
from typing import Sequence

from pluribus_ri.abstraction import (
    compare_postflop_bucket_policies,
    load_abstraction_tables_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze postflop bucket policy calibration metrics for abstraction tuning."
    )
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", default=None)
    parser.add_argument("--board-lengths", default="3,4,5")
    parser.add_argument("--preflop-bucket-policy", choices=["legacy", "canonical169"], default="legacy")
    parser.add_argument(
        "--abstraction-config",
        default=None,
        help="Optional abstraction config JSON; if provided, bucket policies are read from it unless overridden.",
    )
    parser.add_argument("--output-path", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    preflop_bucket_policy = args.preflop_bucket_policy
    policies = _parse_postflop_policies(args.policies)
    if args.abstraction_config:
        tables = load_abstraction_tables_config(args.abstraction_config)
        preflop_bucket_policy = tables.action.preflop_bucket_policy
        if args.policies is None:
            policies = (tables.action.postflop_bucket_policy,)

    board_lengths = _parse_int_tuple(args.board_lengths)
    reports = compare_postflop_bucket_policies(
        policies=policies,
        samples=args.samples,
        seed=args.seed,
        board_lengths=board_lengths,
        preflop_bucket_policy=preflop_bucket_policy,
    )

    payload = {
        "samples": args.samples,
        "seed": args.seed,
        "board_lengths": list(board_lengths),
        "preflop_bucket_policy": preflop_bucket_policy,
        "reports": {name: report.to_dict() for name, report in reports.items()},
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_postflop_policies(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ("legacy", "texture_v1")
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not values:
        return ("legacy", "texture_v1")
    for value in values:
        if value not in {"legacy", "texture_v1"}:
            raise ValueError(f"unsupported postflop policy: {value}")
    return values


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("at least one integer value is required")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
