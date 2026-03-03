from dataclasses import dataclass, replace
import json
from pathlib import Path
import pickle
import random
from statistics import mean, stdev
from typing import Any, Literal, Mapping, Protocol, Sequence

import eval7

from pluribus_ri.core import NoLimitHoldemEngine, Street
from pluribus_ri.solver import (
    NLTHAbstractAction,
    NLTHAbstractGameFactory,
    NLTHAbstractGameState,
    NLTHActionAbstractionConfig,
    NLTHGameConfig,
)

from .policy import BlueprintPolicy


BaselineStrategy = Literal[
    "uniform",
    "check_fold",
    "call_biased",
    "raise_biased",
    "tight_aggressive",
    "loose_aggressive",
    "pot_odds",
    "cheater_weak",
    "cheater_strong",
]
SUPPORTED_BASELINES: tuple[BaselineStrategy, ...] = (
    "uniform",
    "check_fold",
    "call_biased",
    "raise_biased",
    "tight_aggressive",
    "loose_aggressive",
    "pot_odds",
    "cheater_weak",
    "cheater_strong",
)


class _ActionPolicy(Protocol):
    def action_distribution(
        self,
        state: NLTHAbstractGameState,
    ) -> tuple[list[NLTHAbstractAction], Sequence[float]]: ...

    def select_action(
        self,
        state: NLTHAbstractGameState,
        rng: random.Random | None = None,
    ) -> NLTHAbstractAction: ...


@dataclass(frozen=True)
class AIVATConfig:
    """
    AIVAT-style action correction configuration.

    This is a CPU-first implementation that estimates per-decision action values
    with short rollouts and applies the standard action-correction term:
    `q_hat(a_taken) - E_pi[q_hat(a)]`.
    """

    rollout_count_per_action: int = 1
    max_actions_per_rollout: int = 64
    max_branching_for_correction: int = 8
    include_opponent_decisions: bool = True


@dataclass(frozen=True)
class LeagueEvaluationConfig:
    """Configuration for Phase 5 one-vs-field blueprint league evaluation."""

    num_hands_per_seat: int = 24
    random_seed: int = 0
    confidence_z: float = 1.96
    sample_actions: bool = True
    max_actions_per_hand: int = 500
    control_variate_baseline: BaselineStrategy | None = None
    aivat_config: AIVATConfig | None = None


@dataclass(frozen=True)
class ExploitabilityProxyConfig:
    """Approximate exploitability using a fixed baseline strategy pool."""

    num_hands_per_seat: int = 24
    random_seed: int = 0
    confidence_z: float = 1.96
    sample_actions: bool = True
    max_actions_per_hand: int = 500
    baseline_policies: tuple[BaselineStrategy, ...] = SUPPORTED_BASELINES
    control_variate_baseline: BaselineStrategy | None = None
    aivat_config: AIVATConfig | None = None


@dataclass(frozen=True)
class _HeuristicPolicy:
    strategy: BaselineStrategy

    def action_distribution(
        self,
        state: NLTHAbstractGameState,
    ) -> tuple[list[NLTHAbstractAction], list[float]]:
        actions = list(state.legal_actions())
        if not actions:
            raise ValueError("state has no legal actions")
        if self.strategy == "uniform":
            uniform = 1.0 / len(actions)
            return actions, [uniform for _ in actions]

        if self.strategy in {"tight_aggressive", "loose_aggressive", "pot_odds"}:
            return actions, _build_strength_baseline_distribution(
                state=state,
                actions=actions,
                strategy=self.strategy,
            )
        if self.strategy in {"cheater_weak", "cheater_strong"}:
            return actions, _build_cheater_baseline_distribution(
                state=state,
                actions=actions,
                strategy=self.strategy,
            )

        selected = _select_baseline_action(actions=actions, strategy=self.strategy, rng=None)
        return actions, [1.0 if action == selected else 0.0 for action in actions]

    def select_action(
        self,
        state: NLTHAbstractGameState,
        rng: random.Random | None = None,
    ) -> NLTHAbstractAction:
        actions, probs = self.action_distribution(state)
        if rng is None:
            best_idx = max(range(len(probs)), key=lambda idx: probs[idx])
            return actions[best_idx]
        sampled_idx = _sample_index(rng, _normalize_distribution(probs, len(actions)))
        return actions[sampled_idx]


@dataclass(frozen=True)
class _LineupSampleResult:
    samples: list[float]
    max_zero_sum_error: float
    control_variate_reference_samples: list[float] | None = None
    control_variate_max_zero_sum_error: float = 0.0
    aivat_adjusted_samples: list[float] | None = None
    aivat_corrections_applied: int = 0
    aivat_corrections_skipped_branching: int = 0


@dataclass(frozen=True)
class _ActionSlots:
    fold_idx: int | None
    check_idx: int | None
    call_idx: int | None
    raise_idxs: tuple[int, ...]


@dataclass(frozen=True)
class _StrengthContext:
    strength: float
    to_call: int
    pot_odds: float
    stack_pressure: float


_RANK_TO_VALUE = {rank: idx for idx, rank in enumerate("23456789TJQKA", start=2)}
_HANDTYPE_STRENGTH: dict[str, float] = {
    "High Card": 0.26,
    "Pair": 0.44,
    "Two Pair": 0.64,
    "Trips": 0.75,
    "Straight": 0.84,
    "Flush": 0.88,
    "Full House": 0.94,
    "Quads": 0.98,
    "Straight Flush": 0.995,
}
_CARD_CACHE: dict[str, eval7.Card] = {}
_BLUEPRINT_CACHE_VERSION = 1
_BLUEPRINT_CACHE_SUFFIX = ".cache.pkl"


def load_blueprint_policy(path: str | Path) -> BlueprintPolicy:
    blueprint_path = Path(path)
    policy = _load_blueprint_policy_cache(blueprint_path)
    if policy is not None:
        return policy

    with blueprint_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("blueprint payload must be a JSON object")

    policy = BlueprintPolicy.from_snapshot_payload(payload)
    _write_blueprint_policy_cache(blueprint_path, policy)
    return policy


def _blueprint_cache_path(blueprint_path: Path) -> Path:
    return blueprint_path.with_suffix(f"{blueprint_path.suffix}{_BLUEPRINT_CACHE_SUFFIX}")


def _load_blueprint_policy_cache(blueprint_path: Path) -> BlueprintPolicy | None:
    cache_path = _blueprint_cache_path(blueprint_path)
    try:
        source_stat = blueprint_path.stat()
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
    except (FileNotFoundError, OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None

    if not isinstance(cached, dict):
        return None
    if cached.get("version") != _BLUEPRINT_CACHE_VERSION:
        return None
    if cached.get("source_size") != source_stat.st_size:
        return None
    if cached.get("source_mtime_ns") != source_stat.st_mtime_ns:
        return None

    policy = cached.get("policy")
    if not isinstance(policy, BlueprintPolicy):
        return None
    return policy


def _write_blueprint_policy_cache(blueprint_path: Path, policy: BlueprintPolicy) -> None:
    cache_path = _blueprint_cache_path(blueprint_path)
    tmp_cache_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")

    try:
        source_stat = blueprint_path.stat()
        with tmp_cache_path.open("wb") as f:
            pickle.dump(
                {
                    "version": _BLUEPRINT_CACHE_VERSION,
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "policy": policy,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_cache_path.replace(cache_path)
    except OSError:
        try:
            if tmp_cache_path.exists():
                tmp_cache_path.unlink()
        except OSError:
            pass


def run_one_vs_field_league(
    policies: Mapping[str, _ActionPolicy],
    config: LeagueEvaluationConfig | None = None,
    *,
    game_config: NLTHGameConfig | None = None,
    abstraction_config: NLTHActionAbstractionConfig | None = None,
) -> dict[str, Any]:
    eval_config = config or LeagueEvaluationConfig()
    _validate_league_inputs(policies, eval_config)

    entries = list(policies.items())
    base_game_config = game_config or NLTHGameConfig(random_seed=eval_config.random_seed)
    abstraction = abstraction_config or NLTHActionAbstractionConfig()

    matchups: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, float]] = {name: {} for name, _ in entries}

    for candidate_idx, (candidate_name, candidate_policy) in enumerate(entries):
        for field_idx, (field_name, field_policy) in enumerate(entries):
            if candidate_name == field_name:
                continue

            candidate_samples: list[float] = []
            cv_reference_samples: list[float] = []
            aivat_adjusted_samples: list[float] = []
            aivat_corrections_applied = 0
            aivat_corrections_skipped_branching = 0
            max_zero_sum_error = 0.0

            for seat in range(base_game_config.num_players):
                match_seed = (
                    eval_config.random_seed
                    + candidate_idx * 1_000_000
                    + field_idx * 10_000
                    + seat
                )
                lineup_policies = [field_policy for _ in range(base_game_config.num_players)]
                lineup_policies[seat] = candidate_policy

                cv_reference_lineup: list[_ActionPolicy] | None = None
                if eval_config.control_variate_baseline is not None:
                    cv_reference_policy = _HeuristicPolicy(strategy=eval_config.control_variate_baseline)
                    cv_reference_lineup = [field_policy for _ in range(base_game_config.num_players)]
                    cv_reference_lineup[seat] = cv_reference_policy

                lineup_result = _run_lineup_samples(
                    lineup_policies=lineup_policies,
                    candidate_seat=seat,
                    num_hands=eval_config.num_hands_per_seat,
                    random_seed=match_seed,
                    sample_actions=eval_config.sample_actions,
                    max_actions_per_hand=eval_config.max_actions_per_hand,
                    game_config=base_game_config,
                    abstraction_config=abstraction,
                    control_variate_reference_lineup=cv_reference_lineup,
                    aivat_config=eval_config.aivat_config,
                )
                candidate_samples.extend(lineup_result.samples)
                if lineup_result.control_variate_reference_samples is not None:
                    cv_reference_samples.extend(lineup_result.control_variate_reference_samples)
                if lineup_result.aivat_adjusted_samples is not None:
                    aivat_adjusted_samples.extend(lineup_result.aivat_adjusted_samples)
                aivat_corrections_applied += lineup_result.aivat_corrections_applied
                aivat_corrections_skipped_branching += lineup_result.aivat_corrections_skipped_branching
                max_zero_sum_error = max(
                    max_zero_sum_error,
                    lineup_result.max_zero_sum_error,
                    lineup_result.control_variate_max_zero_sum_error,
                )

            summary = _summarize_samples(
                samples=candidate_samples,
                big_blind=base_game_config.big_blind,
                confidence_z=eval_config.confidence_z,
            )
            matchup: dict[str, Any] = {
                "candidate": candidate_name,
                "field": field_name,
                "hands_played": len(candidate_samples),
                "sample_actions": eval_config.sample_actions,
                "mean_utility_per_hand": summary["mean_utility_per_hand"],
                "stdev_utility_per_hand": summary["stdev_utility_per_hand"],
                "ci95_utility_per_hand": summary["ci95_utility_per_hand"],
                "bb_per_hand": summary["bb_per_hand"],
                "mbb_per_hand": summary["mbb_per_hand"],
                "ci95_mbb_per_hand": summary["ci95_mbb_per_hand"],
                "zero_sum_max_abs_error": max_zero_sum_error,
            }
            if eval_config.control_variate_baseline is not None and cv_reference_samples:
                matchup["control_variate"] = _build_control_variate_summary(
                    raw_samples=candidate_samples,
                    reference_samples=cv_reference_samples,
                    baseline_strategy=eval_config.control_variate_baseline,
                    big_blind=base_game_config.big_blind,
                    confidence_z=eval_config.confidence_z,
                )
            if eval_config.aivat_config is not None and aivat_adjusted_samples:
                matchup["aivat"] = _build_aivat_summary(
                    raw_samples=candidate_samples,
                    adjusted_samples=aivat_adjusted_samples,
                    big_blind=base_game_config.big_blind,
                    confidence_z=eval_config.confidence_z,
                    corrections_applied=aivat_corrections_applied,
                    corrections_skipped_branching=aivat_corrections_skipped_branching,
                )

            matchups.append(matchup)
            matrix[candidate_name][field_name] = float(matchup["mbb_per_hand"])

    aggregates = _build_aggregates(matchups, entries)
    return {
        "format": "one_vs_field",
        "num_players": base_game_config.num_players,
        "big_blind": base_game_config.big_blind,
        "num_hands_per_seat": eval_config.num_hands_per_seat,
        "total_hands_per_matchup": eval_config.num_hands_per_seat * base_game_config.num_players,
        "sample_actions": eval_config.sample_actions,
        "confidence_z": eval_config.confidence_z,
        "control_variate_baseline": eval_config.control_variate_baseline,
        "aivat_config": _serialize_aivat_config(eval_config.aivat_config),
        "matchups": matchups,
        "matrix_mbb_per_hand": matrix,
        "aggregates": aggregates,
    }


def run_exploitability_proxy_report(
    policies: Mapping[str, BlueprintPolicy],
    config: ExploitabilityProxyConfig | None = None,
    *,
    game_config: NLTHGameConfig | None = None,
    abstraction_config: NLTHActionAbstractionConfig | None = None,
) -> dict[str, Any]:
    eval_config = config or ExploitabilityProxyConfig()
    _validate_proxy_inputs(policies, eval_config)

    entries = list(policies.items())
    base_game_config = game_config or NLTHGameConfig(random_seed=eval_config.random_seed)
    abstraction = abstraction_config or NLTHActionAbstractionConfig()

    candidates: list[dict[str, Any]] = []
    for candidate_idx, (candidate_name, candidate_policy) in enumerate(entries):
        baseline_matchups: list[dict[str, Any]] = []
        for baseline_idx, baseline_name in enumerate(eval_config.baseline_policies):
            baseline_policy = _HeuristicPolicy(strategy=baseline_name)
            candidate_samples: list[float] = []
            cv_reference_samples: list[float] = []
            aivat_adjusted_samples: list[float] = []
            aivat_corrections_applied = 0
            aivat_corrections_skipped_branching = 0
            max_zero_sum_error = 0.0

            for seat in range(base_game_config.num_players):
                match_seed = (
                    eval_config.random_seed
                    + candidate_idx * 1_000_000
                    + baseline_idx * 10_000
                    + seat
                )
                lineup_policies: list[_ActionPolicy] = [
                    baseline_policy for _ in range(base_game_config.num_players)
                ]
                lineup_policies[seat] = candidate_policy

                cv_reference_lineup: list[_ActionPolicy] | None = None
                if eval_config.control_variate_baseline is not None:
                    cv_reference_policy = _HeuristicPolicy(strategy=eval_config.control_variate_baseline)
                    cv_reference_lineup = [baseline_policy for _ in range(base_game_config.num_players)]
                    cv_reference_lineup[seat] = cv_reference_policy

                lineup_result = _run_lineup_samples(
                    lineup_policies=lineup_policies,
                    candidate_seat=seat,
                    num_hands=eval_config.num_hands_per_seat,
                    random_seed=match_seed,
                    sample_actions=eval_config.sample_actions,
                    max_actions_per_hand=eval_config.max_actions_per_hand,
                    game_config=base_game_config,
                    abstraction_config=abstraction,
                    control_variate_reference_lineup=cv_reference_lineup,
                    aivat_config=eval_config.aivat_config,
                )
                candidate_samples.extend(lineup_result.samples)
                if lineup_result.control_variate_reference_samples is not None:
                    cv_reference_samples.extend(lineup_result.control_variate_reference_samples)
                if lineup_result.aivat_adjusted_samples is not None:
                    aivat_adjusted_samples.extend(lineup_result.aivat_adjusted_samples)
                aivat_corrections_applied += lineup_result.aivat_corrections_applied
                aivat_corrections_skipped_branching += lineup_result.aivat_corrections_skipped_branching
                max_zero_sum_error = max(
                    max_zero_sum_error,
                    lineup_result.max_zero_sum_error,
                    lineup_result.control_variate_max_zero_sum_error,
                )

            summary = _summarize_samples(
                samples=candidate_samples,
                big_blind=base_game_config.big_blind,
                confidence_z=eval_config.confidence_z,
            )
            baseline_result: dict[str, Any] = {
                "baseline": baseline_name,
                "hands_played": len(candidate_samples),
                "sample_actions": eval_config.sample_actions,
                "mean_utility_per_hand": summary["mean_utility_per_hand"],
                "stdev_utility_per_hand": summary["stdev_utility_per_hand"],
                "ci95_utility_per_hand": summary["ci95_utility_per_hand"],
                "bb_per_hand": summary["bb_per_hand"],
                "mbb_per_hand": summary["mbb_per_hand"],
                "ci95_mbb_per_hand": summary["ci95_mbb_per_hand"],
                "zero_sum_max_abs_error": max_zero_sum_error,
            }
            if eval_config.control_variate_baseline is not None and cv_reference_samples:
                baseline_result["control_variate"] = _build_control_variate_summary(
                    raw_samples=candidate_samples,
                    reference_samples=cv_reference_samples,
                    baseline_strategy=eval_config.control_variate_baseline,
                    big_blind=base_game_config.big_blind,
                    confidence_z=eval_config.confidence_z,
                )
            if eval_config.aivat_config is not None and aivat_adjusted_samples:
                baseline_result["aivat"] = _build_aivat_summary(
                    raw_samples=candidate_samples,
                    adjusted_samples=aivat_adjusted_samples,
                    big_blind=base_game_config.big_blind,
                    confidence_z=eval_config.confidence_z,
                    corrections_applied=aivat_corrections_applied,
                    corrections_skipped_branching=aivat_corrections_skipped_branching,
                )
            baseline_matchups.append(baseline_result)

        baseline_mbb = [float(item["mbb_per_hand"]) for item in baseline_matchups]
        worst_case_mbb = min(baseline_mbb)
        proxy_exploitability = max(0.0, -worst_case_mbb)
        candidates.append(
            {
                "candidate": candidate_name,
                "baseline_matchups": baseline_matchups,
                "hands_played": sum(int(item["hands_played"]) for item in baseline_matchups),
                "mean_mbb_per_hand_vs_baselines": float(mean(baseline_mbb)),
                "worst_case_mbb_per_hand": float(worst_case_mbb),
                "proxy_exploitability_mbb_per_hand": float(proxy_exploitability),
            }
        )

    ranking = [
        {
            "candidate": item["candidate"],
            "proxy_exploitability_mbb_per_hand": item["proxy_exploitability_mbb_per_hand"],
        }
        for item in sorted(candidates, key=lambda entry: entry["proxy_exploitability_mbb_per_hand"])
    ]
    return {
        "format": "exploitability_proxy",
        "num_players": base_game_config.num_players,
        "big_blind": base_game_config.big_blind,
        "num_hands_per_seat": eval_config.num_hands_per_seat,
        "total_hands_per_baseline": eval_config.num_hands_per_seat * base_game_config.num_players,
        "sample_actions": eval_config.sample_actions,
        "confidence_z": eval_config.confidence_z,
        "control_variate_baseline": eval_config.control_variate_baseline,
        "aivat_config": _serialize_aivat_config(eval_config.aivat_config),
        "baseline_policies": list(eval_config.baseline_policies),
        "candidates": candidates,
        "ranking_by_proxy_exploitability": ranking,
    }


def _run_lineup_samples(
    *,
    lineup_policies: Sequence[_ActionPolicy],
    candidate_seat: int,
    num_hands: int,
    random_seed: int,
    sample_actions: bool,
    max_actions_per_hand: int,
    game_config: NLTHGameConfig,
    abstraction_config: NLTHActionAbstractionConfig,
    control_variate_reference_lineup: Sequence[_ActionPolicy] | None = None,
    aivat_config: AIVATConfig | None = None,
) -> _LineupSampleResult:
    factory = NLTHAbstractGameFactory(
        game_config=replace(game_config, random_seed=random_seed),
        abstraction_config=abstraction_config,
    )
    reference_factory: NLTHAbstractGameFactory | None = None
    if control_variate_reference_lineup is not None:
        reference_factory = NLTHAbstractGameFactory(
            game_config=replace(game_config, random_seed=random_seed),
            abstraction_config=abstraction_config,
        )

    action_rng = random.Random(random_seed ^ 0x9E3779B1)
    reference_action_rng = random.Random(random_seed ^ 0x85EBCA77)
    aivat_rng = random.Random(random_seed ^ 0xC2B2AE35)

    samples: list[float] = []
    max_zero_sum_error = 0.0
    control_samples: list[float] | None = [] if reference_factory is not None else None
    control_max_zero_sum_error = 0.0
    aivat_samples: list[float] | None = (
        [] if aivat_config is not None and sample_actions else None
    )
    aivat_corrections_applied = 0
    aivat_corrections_skipped_branching = 0

    for _ in range(num_hands):
        state = factory.root_state()
        if aivat_samples is not None and aivat_config is not None:
            terminal_state, correction, applied, skipped_branching = _rollout_lineup_with_aivat(
                state=state,
                lineup_policies=lineup_policies,
                candidate_seat=candidate_seat,
                rng=action_rng,
                aivat_rng=aivat_rng,
                max_actions_per_hand=max_actions_per_hand,
                aivat_config=aivat_config,
            )
            aivat_corrections_applied += applied
            aivat_corrections_skipped_branching += skipped_branching
        else:
            terminal_state = _rollout_lineup(
                state=state,
                lineup_policies=lineup_policies,
                rng=action_rng,
                sample_actions=sample_actions,
                max_actions_per_hand=max_actions_per_hand,
            )
            correction = 0.0

        seat_utilities = [terminal_state.utility(seat) for seat in range(game_config.num_players)]
        raw_utility = seat_utilities[candidate_seat]
        samples.append(raw_utility)
        max_zero_sum_error = max(max_zero_sum_error, abs(sum(seat_utilities)))
        if aivat_samples is not None:
            aivat_samples.append(raw_utility - correction)

        if reference_factory is not None and control_samples is not None and control_variate_reference_lineup is not None:
            reference_state = reference_factory.root_state()
            terminal_reference = _rollout_lineup(
                state=reference_state,
                lineup_policies=control_variate_reference_lineup,
                rng=reference_action_rng,
                sample_actions=sample_actions,
                max_actions_per_hand=max_actions_per_hand,
            )
            reference_utilities = [terminal_reference.utility(seat) for seat in range(game_config.num_players)]
            control_samples.append(reference_utilities[candidate_seat])
            control_max_zero_sum_error = max(control_max_zero_sum_error, abs(sum(reference_utilities)))

    return _LineupSampleResult(
        samples=samples,
        max_zero_sum_error=max_zero_sum_error,
        control_variate_reference_samples=control_samples,
        control_variate_max_zero_sum_error=control_max_zero_sum_error,
        aivat_adjusted_samples=aivat_samples,
        aivat_corrections_applied=aivat_corrections_applied,
        aivat_corrections_skipped_branching=aivat_corrections_skipped_branching,
    )


def _rollout_lineup(
    *,
    state: NLTHAbstractGameState,
    lineup_policies: Sequence[_ActionPolicy],
    rng: random.Random,
    sample_actions: bool,
    max_actions_per_hand: int,
    allow_truncation: bool = False,
) -> NLTHAbstractGameState:
    guard = 0
    current = state
    while not current.is_terminal():
        guard += 1
        if guard > max_actions_per_hand:
            if allow_truncation:
                return current
            raise RuntimeError("evaluation hand exceeded action guard")
        acting_seat = current.current_player()
        policy = lineup_policies[acting_seat]
        selected, _actions, _probs, _idx = _sample_policy_action(
            policy=policy,
            state=current,
            rng=rng,
            sample_actions=sample_actions,
        )
        current = current.child(selected)
    return current


def _rollout_lineup_with_aivat(
    *,
    state: NLTHAbstractGameState,
    lineup_policies: Sequence[_ActionPolicy],
    candidate_seat: int,
    rng: random.Random,
    aivat_rng: random.Random,
    max_actions_per_hand: int,
    aivat_config: AIVATConfig,
) -> tuple[NLTHAbstractGameState, float, int, int]:
    guard = 0
    current = state
    correction_sum = 0.0
    corrections_applied = 0
    corrections_skipped_branching = 0

    while not current.is_terminal():
        guard += 1
        if guard > max_actions_per_hand:
            raise RuntimeError("evaluation hand exceeded action guard")
        acting_seat = current.current_player()
        policy = lineup_policies[acting_seat]
        selected, actions, probs, selected_idx = _sample_policy_action(
            policy=policy,
            state=current,
            rng=rng,
            sample_actions=True,
        )

        should_correct = aivat_config.include_opponent_decisions or acting_seat == candidate_seat
        if should_correct and len(actions) > 1:
            if len(actions) <= aivat_config.max_branching_for_correction:
                q_values = _estimate_aivat_action_values(
                    state=current,
                    actions=actions,
                    lineup_policies=lineup_policies,
                    candidate_seat=candidate_seat,
                    rng=aivat_rng,
                    aivat_config=aivat_config,
                )
                expected_q = sum(prob * q_value for prob, q_value in zip(probs, q_values))
                correction_sum += q_values[selected_idx] - expected_q
                corrections_applied += 1
            else:
                corrections_skipped_branching += 1

        current = current.child(selected)

    return current, correction_sum, corrections_applied, corrections_skipped_branching


def _estimate_aivat_action_values(
    *,
    state: NLTHAbstractGameState,
    actions: Sequence[NLTHAbstractAction],
    lineup_policies: Sequence[_ActionPolicy],
    candidate_seat: int,
    rng: random.Random,
    aivat_config: AIVATConfig,
) -> list[float]:
    estimates: list[float] = []
    for action in actions:
        total = 0.0
        for _ in range(aivat_config.rollout_count_per_action):
            child_state = state.child(action)
            terminal = _rollout_lineup(
                state=child_state,
                lineup_policies=lineup_policies,
                rng=rng,
                sample_actions=True,
                max_actions_per_hand=aivat_config.max_actions_per_rollout,
                allow_truncation=True,
            )
            total += _state_player_utility(terminal, candidate_seat)
        estimates.append(total / aivat_config.rollout_count_per_action)
    return estimates


def _state_player_utility(state: NLTHAbstractGameState, player: int) -> float:
    if state.is_terminal():
        return state.utility(player)
    return float(state.engine.stacks[player] - state.root_stacks[player])


def _sample_policy_action(
    *,
    policy: _ActionPolicy,
    state: NLTHAbstractGameState,
    rng: random.Random,
    sample_actions: bool,
) -> tuple[NLTHAbstractAction, list[NLTHAbstractAction], Sequence[float], int]:
    actions, probs = policy.action_distribution(state)
    if not actions:
        raise ValueError("state has no legal actions")
    if isinstance(policy, BlueprintPolicy):
        # BlueprintPolicy already returns clipped, normalized distributions.
        normalized = probs
    else:
        normalized = _normalize_distribution(probs, len(actions))
    if sample_actions:
        idx = _sample_index(rng, normalized)
    else:
        idx = max(range(len(normalized)), key=lambda i: normalized[i])
    return actions[idx], actions, normalized, idx


def _summarize_samples(
    *,
    samples: Sequence[float],
    big_blind: int,
    confidence_z: float,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    if big_blind <= 0:
        raise ValueError("big_blind must be positive")

    sample_mean = float(mean(samples))
    if len(samples) > 1:
        sample_stdev = float(stdev(samples))
        stderr = sample_stdev / (len(samples) ** 0.5)
        ci_low = sample_mean - confidence_z * stderr
        ci_high = sample_mean + confidence_z * stderr
    else:
        sample_stdev = 0.0
        ci_low = sample_mean
        ci_high = sample_mean

    bb_mean = sample_mean / big_blind
    mbb_mean = bb_mean * 1000.0

    return {
        "mean_utility_per_hand": sample_mean,
        "stdev_utility_per_hand": sample_stdev,
        "ci95_utility_per_hand": [ci_low, ci_high],
        "bb_per_hand": bb_mean,
        "mbb_per_hand": mbb_mean,
        "ci95_mbb_per_hand": [ci_low / big_blind * 1000.0, ci_high / big_blind * 1000.0],
    }


def _build_control_variate_summary(
    *,
    raw_samples: Sequence[float],
    reference_samples: Sequence[float],
    baseline_strategy: BaselineStrategy,
    big_blind: int,
    confidence_z: float,
) -> dict[str, Any]:
    if len(raw_samples) != len(reference_samples):
        raise ValueError("control variate reference samples must match raw sample count")
    if not raw_samples:
        raise ValueError("control variate samples must not be empty")

    raw_summary = _summarize_samples(samples=raw_samples, big_blind=big_blind, confidence_z=confidence_z)
    reference_summary = _summarize_samples(samples=reference_samples, big_blind=big_blind, confidence_z=confidence_z)

    beta = _estimate_beta(raw_samples=raw_samples, reference_samples=reference_samples)
    reference_mean = float(mean(reference_samples))
    adjusted_samples = [
        raw_sample - beta * (reference_sample - reference_mean)
        for raw_sample, reference_sample in zip(raw_samples, reference_samples)
    ]
    adjusted_summary = _summarize_samples(samples=adjusted_samples, big_blind=big_blind, confidence_z=confidence_z)

    raw_ci_width = _ci_width(raw_summary)
    adjusted_ci_width = _ci_width(adjusted_summary)
    raw_stdev = float(raw_summary["stdev_utility_per_hand"])
    adjusted_stdev = float(adjusted_summary["stdev_utility_per_hand"])

    return {
        "baseline_strategy": baseline_strategy,
        "beta": float(beta),
        "reference_mean_utility_per_hand": reference_summary["mean_utility_per_hand"],
        "reference_stdev_utility_per_hand": reference_summary["stdev_utility_per_hand"],
        "adjusted_mean_utility_per_hand": adjusted_summary["mean_utility_per_hand"],
        "adjusted_stdev_utility_per_hand": adjusted_summary["stdev_utility_per_hand"],
        "adjusted_ci95_utility_per_hand": adjusted_summary["ci95_utility_per_hand"],
        "adjusted_mbb_per_hand": adjusted_summary["mbb_per_hand"],
        "adjusted_ci95_mbb_per_hand": adjusted_summary["ci95_mbb_per_hand"],
        "raw_ci95_width_utility_per_hand": raw_ci_width,
        "adjusted_ci95_width_utility_per_hand": adjusted_ci_width,
        "stdev_reduction_pct": _reduction_pct(raw_stdev, adjusted_stdev),
        "ci95_width_reduction_pct": _reduction_pct(raw_ci_width, adjusted_ci_width),
    }


def _build_aivat_summary(
    *,
    raw_samples: Sequence[float],
    adjusted_samples: Sequence[float],
    big_blind: int,
    confidence_z: float,
    corrections_applied: int,
    corrections_skipped_branching: int,
) -> dict[str, Any]:
    if len(raw_samples) != len(adjusted_samples):
        raise ValueError("aivat adjusted samples must match raw sample count")
    if not raw_samples:
        raise ValueError("aivat samples must not be empty")

    raw_summary = _summarize_samples(samples=raw_samples, big_blind=big_blind, confidence_z=confidence_z)
    adjusted_summary = _summarize_samples(samples=adjusted_samples, big_blind=big_blind, confidence_z=confidence_z)

    raw_ci_width = _ci_width(raw_summary)
    adjusted_ci_width = _ci_width(adjusted_summary)
    raw_stdev = float(raw_summary["stdev_utility_per_hand"])
    adjusted_stdev = float(adjusted_summary["stdev_utility_per_hand"])

    return {
        "method": "action_correction_only",
        "adjusted_mean_utility_per_hand": adjusted_summary["mean_utility_per_hand"],
        "adjusted_stdev_utility_per_hand": adjusted_summary["stdev_utility_per_hand"],
        "adjusted_ci95_utility_per_hand": adjusted_summary["ci95_utility_per_hand"],
        "adjusted_mbb_per_hand": adjusted_summary["mbb_per_hand"],
        "adjusted_ci95_mbb_per_hand": adjusted_summary["ci95_mbb_per_hand"],
        "raw_ci95_width_utility_per_hand": raw_ci_width,
        "adjusted_ci95_width_utility_per_hand": adjusted_ci_width,
        "stdev_reduction_pct": _reduction_pct(raw_stdev, adjusted_stdev),
        "ci95_width_reduction_pct": _reduction_pct(raw_ci_width, adjusted_ci_width),
        "corrections_applied": corrections_applied,
        "corrections_skipped_branching": corrections_skipped_branching,
    }


def _build_aggregates(
    matchups: Sequence[dict[str, Any]],
    entries: Sequence[tuple[str, _ActionPolicy]],
) -> dict[str, dict[str, float | int]]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for candidate_name, _ in entries:
        candidate_matchups = [m for m in matchups if m["candidate"] == candidate_name]
        if not candidate_matchups:
            continue
        mbb_values = [float(m["mbb_per_hand"]) for m in candidate_matchups]
        ci_lows = [float(m["ci95_mbb_per_hand"][0]) for m in candidate_matchups]
        ci_highs = [float(m["ci95_mbb_per_hand"][1]) for m in candidate_matchups]
        hands = sum(int(m["hands_played"]) for m in candidate_matchups)
        aggregates[candidate_name] = {
            "mean_mbb_per_hand_vs_field": float(mean(mbb_values)),
            "min_ci95_low_mbb_per_hand": float(min(ci_lows)),
            "max_ci95_high_mbb_per_hand": float(max(ci_highs)),
            "hands_played": hands,
        }
    return aggregates


def _select_baseline_action(
    *,
    actions: Sequence[NLTHAbstractAction],
    strategy: BaselineStrategy,
    rng: random.Random | None,
) -> NLTHAbstractAction:
    if strategy == "uniform":
        if rng is None:
            return actions[0]
        return rng.choice(list(actions))

    check_action = _first_action(actions, kind="check")
    fold_action = _first_action(actions, kind="fold")
    call_action = _first_action(actions, kind="call")
    raise_actions = [action for action in actions if action.kind == "raise"]

    if strategy == "check_fold":
        return check_action or fold_action or call_action or actions[0]
    if strategy == "call_biased":
        return call_action or check_action or _min_raise(raise_actions) or fold_action or actions[0]
    if strategy == "raise_biased":
        return _max_raise(raise_actions) or call_action or check_action or fold_action or actions[0]

    raise ValueError(f"unsupported baseline strategy: {strategy}")


def _build_strength_baseline_distribution(
    *,
    state: NLTHAbstractGameState,
    actions: Sequence[NLTHAbstractAction],
    strategy: BaselineStrategy,
) -> list[float]:
    seat = state.current_player()
    engine = state.engine
    player = engine.players[seat]
    hole_cards = player.hole_cards
    if hole_cards is None:
        uniform = 1.0 / len(actions)
        return [uniform for _ in actions]

    context = _build_strength_context(state=state, hole_cards=hole_cards)
    slots = _classify_action_slots(actions)
    weights = [0.0 for _ in actions]

    if strategy == "tight_aggressive":
        _apply_tight_aggressive(weights=weights, slots=slots, context=context)
    elif strategy == "loose_aggressive":
        _apply_loose_aggressive(weights=weights, slots=slots, context=context)
    elif strategy == "pot_odds":
        _apply_pot_odds(weights=weights, slots=slots, context=context)
    else:
        raise ValueError(f"unsupported baseline strategy: {strategy}")

    return _normalize_distribution(weights, len(actions))


def _build_cheater_baseline_distribution(
    *,
    state: NLTHAbstractGameState,
    actions: Sequence[NLTHAbstractAction],
    strategy: BaselineStrategy,
) -> list[float]:
    seat = state.current_player()
    engine = state.engine
    player = engine.players[seat]
    hole_cards = player.hole_cards
    if hole_cards is None:
        uniform = 1.0 / len(actions)
        return [uniform for _ in actions]

    context = _build_strength_context(state=state, hole_cards=hole_cards)
    slots = _classify_action_slots(actions)
    weights = [0.0 for _ in actions]
    cheating_signal = _estimate_cheater_signal(state=state, seat=seat, strategy=strategy)

    if strategy == "cheater_weak":
        _apply_cheater_weak(
            weights=weights,
            slots=slots,
            context=context,
            cheating_signal=cheating_signal,
        )
    elif strategy == "cheater_strong":
        _apply_cheater_strong(
            weights=weights,
            slots=slots,
            context=context,
            cheating_signal=cheating_signal,
        )
    else:
        raise ValueError(f"unsupported baseline strategy: {strategy}")

    return _normalize_distribution(weights, len(actions))


def _estimate_cheater_signal(
    *,
    state: NLTHAbstractGameState,
    seat: int,
    strategy: BaselineStrategy,
) -> float:
    if strategy == "cheater_strong":
        return _estimate_strong_cheater_showdown_share(state=state, seat=seat)
    if strategy == "cheater_weak":
        return _estimate_weak_cheater_advantage(state=state, seat=seat)
    raise ValueError(f"unsupported baseline strategy: {strategy}")


def _estimate_weak_cheater_advantage(
    *,
    state: NLTHAbstractGameState,
    seat: int,
) -> float:
    engine = state.engine
    hero_cards = engine.players[seat].hole_cards
    if hero_cards is None:
        return 0.0
    opponent_holes = _active_opponent_hole_cards(engine=engine, seat=seat)
    if not opponent_holes:
        return 1.0
    board = tuple(engine.board)
    if len(board) >= 5:
        return _showdown_share_on_board(engine=engine, seat=seat, board=board[:5])
    hero_strength = _estimate_hand_strength(
        hole_cards=hero_cards,
        board=board,
        street=engine.street,
    )
    wins = 0.0
    for opponent_cards in opponent_holes:
        opponent_strength = _estimate_hand_strength(
            hole_cards=opponent_cards,
            board=board,
            street=engine.street,
        )
        if hero_strength > opponent_strength + 1e-9:
            wins += 1.0
        elif abs(hero_strength - opponent_strength) <= 1e-9:
            wins += 0.5
    return _clamp(wins / len(opponent_holes))


def _estimate_strong_cheater_showdown_share(
    *,
    state: NLTHAbstractGameState,
    seat: int,
) -> float:
    engine = state.engine
    board = tuple(engine.board)
    if len(board) >= 5:
        final_board = board[:5]
    else:
        missing = 5 - len(board)
        start = int(engine._deck_index)
        future_cards = tuple(engine.deck[start : start + missing])
        if len(future_cards) != missing:
            return _estimate_weak_cheater_advantage(state=state, seat=seat)
        final_board = board + future_cards
    return _showdown_share_on_board(engine=engine, seat=seat, board=final_board)


def _showdown_share_on_board(
    *,
    engine: NoLimitHoldemEngine,
    seat: int,
    board: Sequence[str],
) -> float:
    hero_cards = engine.players[seat].hole_cards
    if hero_cards is None:
        return 0.0
    opponents = [
        opp_seat
        for opp_seat, player in enumerate(engine.players)
        if opp_seat != seat and player.in_hand and player.hole_cards is not None
    ]
    if not opponents:
        return 1.0

    board_cards = [_card_from_cache(card) for card in board]
    hero_score = eval7.evaluate([_card_from_cache(hero_cards[0]), _card_from_cache(hero_cards[1]), *board_cards])
    opponent_scores = [
        eval7.evaluate(
            [
                _card_from_cache(engine.players[opp_seat].hole_cards[0]),  # type: ignore[index]
                _card_from_cache(engine.players[opp_seat].hole_cards[1]),  # type: ignore[index]
                *board_cards,
            ]
        )
        for opp_seat in opponents
    ]
    best_score = max([hero_score, *opponent_scores])
    if hero_score < best_score:
        return 0.0
    tied_winners = 1 + sum(1 for score in opponent_scores if score == hero_score)
    return 1.0 / tied_winners


def _active_opponent_hole_cards(
    *,
    engine: NoLimitHoldemEngine,
    seat: int,
) -> list[tuple[str, str]]:
    holes: list[tuple[str, str]] = []
    for opp_seat, player in enumerate(engine.players):
        if opp_seat == seat:
            continue
        if not player.in_hand or player.hole_cards is None:
            continue
        holes.append(player.hole_cards)
    return holes


def _apply_cheater_weak(
    *,
    weights: list[float],
    slots: _ActionSlots,
    context: _StrengthContext,
    cheating_signal: float,
) -> None:
    required_call = 0.24 + (0.52 * context.pot_odds) + (0.20 * context.stack_pressure)
    if context.to_call == 0:
        if cheating_signal >= 0.74:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.58)
            _add_check_weight(weights=weights, slots=slots, weight=0.42)
        elif cheating_signal >= 0.58:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.32)
            _add_check_weight(weights=weights, slots=slots, weight=0.68)
        else:
            _add_check_weight(weights=weights, slots=slots, weight=1.0)
    else:
        if cheating_signal >= 0.84:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.52)
            _add_call_weight(weights=weights, slots=slots, weight=0.48)
        elif cheating_signal >= required_call + 0.10:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.20)
            _add_call_weight(weights=weights, slots=slots, weight=0.80)
        elif cheating_signal >= required_call:
            _add_call_weight(weights=weights, slots=slots, weight=0.84)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.16)
        elif cheating_signal + 0.08 >= required_call and context.pot_odds <= 0.12:
            _add_call_weight(weights=weights, slots=slots, weight=0.36)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.64)
        else:
            _add_defensive_weight(weights=weights, slots=slots, weight=1.0)


def _apply_cheater_strong(
    *,
    weights: list[float],
    slots: _ActionSlots,
    context: _StrengthContext,
    cheating_signal: float,
) -> None:
    required_call = context.pot_odds + (0.04 * context.stack_pressure)
    if context.to_call == 0:
        if cheating_signal >= 0.92:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.86)
            _add_check_weight(weights=weights, slots=slots, weight=0.14)
        elif cheating_signal >= 0.76:
            _add_raise_weight(weights=weights, slots=slots, level="medium", weight=0.66)
            _add_check_weight(weights=weights, slots=slots, weight=0.34)
        elif cheating_signal >= 0.58:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.35)
            _add_check_weight(weights=weights, slots=slots, weight=0.65)
        else:
            _add_check_weight(weights=weights, slots=slots, weight=1.0)
    else:
        if cheating_signal >= 0.98:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.92)
            _add_call_weight(weights=weights, slots=slots, weight=0.08)
        elif cheating_signal >= 0.82:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.58)
            _add_call_weight(weights=weights, slots=slots, weight=0.42)
        elif cheating_signal >= required_call + 0.12:
            _add_raise_weight(weights=weights, slots=slots, level="medium", weight=0.45)
            _add_call_weight(weights=weights, slots=slots, weight=0.55)
        elif cheating_signal >= required_call:
            _add_call_weight(weights=weights, slots=slots, weight=0.95)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.05)
        else:
            _add_defensive_weight(weights=weights, slots=slots, weight=1.0)


def _build_strength_context(
    *,
    state: NLTHAbstractGameState,
    hole_cards: tuple[str, str],
) -> _StrengthContext:
    engine = state.engine
    seat = state.current_player()
    player = engine.players[seat]
    to_call = max(0, engine.current_bet - player.contributed_street)
    pot = max(1, engine.total_pot)
    pot_odds = to_call / max(1, pot + to_call)
    stack_pressure = to_call / max(1, player.stack)
    strength = _estimate_hand_strength(
        hole_cards=hole_cards,
        board=tuple(engine.board),
        street=engine.street,
    )
    return _StrengthContext(
        strength=_clamp(strength),
        to_call=to_call,
        pot_odds=pot_odds,
        stack_pressure=stack_pressure,
    )


def _apply_tight_aggressive(
    *,
    weights: list[float],
    slots: _ActionSlots,
    context: _StrengthContext,
) -> None:
    required_call = 0.36 + (0.45 * context.pot_odds) + (0.20 * context.stack_pressure)
    if context.to_call == 0:
        if context.strength >= 0.80:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.72)
            _add_check_weight(weights=weights, slots=slots, weight=0.28)
        elif context.strength >= 0.64:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.55)
            _add_check_weight(weights=weights, slots=slots, weight=0.45)
        elif context.strength >= 0.48:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.12)
            _add_check_weight(weights=weights, slots=slots, weight=0.88)
        else:
            _add_check_weight(weights=weights, slots=slots, weight=1.0)
    else:
        if context.strength >= 0.84:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.62)
            _add_call_weight(weights=weights, slots=slots, weight=0.38)
        elif context.strength >= 0.68:
            _add_raise_weight(weights=weights, slots=slots, level="medium", weight=0.22)
            _add_call_weight(weights=weights, slots=slots, weight=0.78)
        elif context.strength >= required_call:
            _add_call_weight(weights=weights, slots=slots, weight=0.88)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.12)
        elif context.strength + 0.08 >= required_call and context.pot_odds <= 0.14:
            _add_call_weight(weights=weights, slots=slots, weight=0.45)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.55)
        else:
            _add_defensive_weight(weights=weights, slots=slots, weight=1.0)


def _apply_loose_aggressive(
    *,
    weights: list[float],
    slots: _ActionSlots,
    context: _StrengthContext,
) -> None:
    required_call = 0.24 + (0.34 * context.pot_odds) + (0.14 * context.stack_pressure)
    if context.to_call == 0:
        if context.strength >= 0.76:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.70)
            _add_check_weight(weights=weights, slots=slots, weight=0.30)
        elif context.strength >= 0.54:
            _add_raise_weight(weights=weights, slots=slots, level="medium", weight=0.63)
            _add_check_weight(weights=weights, slots=slots, weight=0.37)
        elif context.strength >= 0.34:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.38)
            _add_check_weight(weights=weights, slots=slots, weight=0.62)
        else:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.18)
            _add_check_weight(weights=weights, slots=slots, weight=0.82)
    else:
        if context.strength >= 0.78:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.58)
            _add_call_weight(weights=weights, slots=slots, weight=0.42)
        elif context.strength >= 0.58:
            _add_raise_weight(weights=weights, slots=slots, level="medium", weight=0.32)
            _add_call_weight(weights=weights, slots=slots, weight=0.68)
        elif context.strength >= required_call:
            _add_call_weight(weights=weights, slots=slots, weight=0.72)
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.16)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.12)
        elif context.pot_odds <= 0.10 and context.strength + 0.12 >= required_call:
            _add_call_weight(weights=weights, slots=slots, weight=0.45)
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.10)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.45)
        else:
            _add_defensive_weight(weights=weights, slots=slots, weight=0.72)
            _add_call_weight(weights=weights, slots=slots, weight=0.20)
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.08)


def _apply_pot_odds(
    *,
    weights: list[float],
    slots: _ActionSlots,
    context: _StrengthContext,
) -> None:
    required_call = 0.18 + (0.70 * context.pot_odds) + (0.25 * context.stack_pressure)
    if context.to_call == 0:
        if context.strength >= 0.86:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.74)
            _add_check_weight(weights=weights, slots=slots, weight=0.26)
        elif context.strength >= 0.70:
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.42)
            _add_check_weight(weights=weights, slots=slots, weight=0.58)
        else:
            _add_check_weight(weights=weights, slots=slots, weight=1.0)
    else:
        if context.strength >= 0.90:
            _add_raise_weight(weights=weights, slots=slots, level="high", weight=0.70)
            _add_call_weight(weights=weights, slots=slots, weight=0.30)
        elif context.strength >= max(required_call, 0.62):
            _add_call_weight(weights=weights, slots=slots, weight=0.86)
            _add_raise_weight(weights=weights, slots=slots, level="low", weight=0.14)
        elif context.strength >= required_call:
            _add_call_weight(weights=weights, slots=slots, weight=0.75)
            _add_defensive_weight(weights=weights, slots=slots, weight=0.25)
        else:
            _add_defensive_weight(weights=weights, slots=slots, weight=1.0)


def _classify_action_slots(actions: Sequence[NLTHAbstractAction]) -> _ActionSlots:
    fold_idx: int | None = None
    check_idx: int | None = None
    call_idx: int | None = None
    raise_idxs: list[int] = []
    for idx, action in enumerate(actions):
        if action.kind == "fold":
            fold_idx = idx
        elif action.kind == "check":
            check_idx = idx
        elif action.kind == "call":
            call_idx = idx
        elif action.kind == "raise":
            raise_idxs.append(idx)
    raise_idxs.sort(key=lambda idx: actions[idx].amount)
    return _ActionSlots(
        fold_idx=fold_idx,
        check_idx=check_idx,
        call_idx=call_idx,
        raise_idxs=tuple(raise_idxs),
    )


def _add_defensive_weight(*, weights: list[float], slots: _ActionSlots, weight: float) -> None:
    if slots.fold_idx is not None:
        weights[slots.fold_idx] += weight
    elif slots.check_idx is not None:
        weights[slots.check_idx] += weight
    elif slots.call_idx is not None:
        weights[slots.call_idx] += weight


def _add_check_weight(*, weights: list[float], slots: _ActionSlots, weight: float) -> None:
    if slots.check_idx is not None:
        weights[slots.check_idx] += weight
    else:
        _add_defensive_weight(weights=weights, slots=slots, weight=weight)


def _add_call_weight(*, weights: list[float], slots: _ActionSlots, weight: float) -> None:
    if slots.call_idx is not None:
        weights[slots.call_idx] += weight
    elif slots.check_idx is not None:
        weights[slots.check_idx] += weight
    elif slots.fold_idx is not None:
        weights[slots.fold_idx] += weight


def _add_raise_weight(
    *,
    weights: list[float],
    slots: _ActionSlots,
    level: Literal["low", "medium", "high"],
    weight: float,
) -> None:
    idx = _select_raise_index(slots.raise_idxs, level)
    if idx is not None:
        weights[idx] += weight
    else:
        _add_call_weight(weights=weights, slots=slots, weight=weight)


def _select_raise_index(
    raise_idxs: Sequence[int],
    level: Literal["low", "medium", "high"],
) -> int | None:
    if not raise_idxs:
        return None
    if level == "low":
        return raise_idxs[0]
    if level == "high":
        return raise_idxs[-1]
    return raise_idxs[len(raise_idxs) // 2]


def _estimate_hand_strength(
    *,
    hole_cards: tuple[str, str],
    board: tuple[str, ...],
    street: Street,
) -> float:
    if street == Street.PREFLOP or len(board) == 0:
        return _estimate_preflop_strength(hole_cards)
    return _estimate_postflop_strength(hole_cards=hole_cards, board=board)


def _estimate_preflop_strength(hole_cards: tuple[str, str]) -> float:
    c1, c2 = hole_cards
    r1 = _RANK_TO_VALUE[c1[0]]
    r2 = _RANK_TO_VALUE[c2[0]]
    high = max(r1, r2)
    low = min(r1, r2)
    suited = c1[1] == c2[1]
    gap = high - low
    if high == low:
        return _clamp(0.55 + ((high - 2) / 12.0) * 0.40)

    score = 0.22 + ((high - 2) / 12.0) * 0.32 + ((low - 2) / 12.0) * 0.14
    if suited:
        score += 0.08
    if gap == 1:
        score += 0.06
    elif gap == 2:
        score += 0.03
    elif gap == 3:
        score += 0.01
    if high >= 13 and low >= 10:
        score += 0.09
    elif high == 14 and low >= 9:
        score += 0.05
    return _clamp(score)


def _estimate_postflop_strength(*, hole_cards: tuple[str, str], board: tuple[str, ...]) -> float:
    cards = [_card_from_cache(card) for card in (*hole_cards, *board)]
    hand_value = eval7.evaluate(cards)
    hand_type = eval7.handtype(hand_value)
    strength = _HANDTYPE_STRENGTH.get(hand_type, 0.25)

    if hand_type == "Pair":
        hole_values = sorted((_RANK_TO_VALUE[hole_cards[0][0]], _RANK_TO_VALUE[hole_cards[1][0]]), reverse=True)
        board_values = sorted((_RANK_TO_VALUE[card[0]] for card in board), reverse=True)
        if board_values:
            if board_values[0] in hole_values:
                strength += 0.06
            if hole_values[0] == hole_values[1] and hole_values[0] > board_values[0]:
                strength += 0.05
            if hole_values[0] > board_values[0] and hole_values[1] > board_values[0]:
                strength += 0.02

    strength += _draw_bonus(hole_cards=hole_cards, board=board, hand_type=hand_type)
    return _clamp(strength)


def _draw_bonus(*, hole_cards: tuple[str, str], board: tuple[str, ...], hand_type: str) -> float:
    if len(board) >= 5:
        return 0.0
    bonus = 0.0
    if hand_type not in {"Flush", "Full House", "Quads", "Straight Flush"}:
        suit_counts: dict[str, int] = {}
        for card in (*hole_cards, *board):
            suit = card[1]
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        max_suit = max(suit_counts.values(), default=0)
        if max_suit >= 4:
            bonus += 0.08 if len(board) == 3 else 0.06

    ranks = {_RANK_TO_VALUE[card[0]] for card in (*hole_cards, *board)}
    if 14 in ranks:
        ranks.add(1)
    best_hits = 0
    for start in range(1, 11):
        target = set(range(start, start + 5))
        hits = len(target.intersection(ranks))
        if hits > best_hits:
            best_hits = hits
    if best_hits >= 4:
        bonus += 0.07
    elif best_hits == 3 and len(board) <= 4:
        bonus += 0.02
    return bonus


def _card_from_cache(card: str) -> eval7.Card:
    cached = _CARD_CACHE.get(card)
    if cached is None:
        cached = eval7.Card(card)
        _CARD_CACHE[card] = cached
    return cached


def _clamp(value: float) -> float:
    return max(0.0, min(0.995, float(value)))


def _first_action(actions: Sequence[NLTHAbstractAction], *, kind: str) -> NLTHAbstractAction | None:
    for action in actions:
        if action.kind == kind:
            return action
    return None


def _min_raise(actions: Sequence[NLTHAbstractAction]) -> NLTHAbstractAction | None:
    if not actions:
        return None
    return min(actions, key=lambda action: action.amount)


def _max_raise(actions: Sequence[NLTHAbstractAction]) -> NLTHAbstractAction | None:
    if not actions:
        return None
    return max(actions, key=lambda action: action.amount)


def _normalize_distribution(probs: Sequence[float], size: int) -> list[float]:
    if len(probs) != size:
        raise ValueError("action/probability length mismatch")
    clipped = [max(0.0, float(value)) for value in probs]
    total = sum(clipped)
    if total <= 0.0:
        uniform = 1.0 / size
        return [uniform for _ in range(size)]
    return [value / total for value in clipped]


def _sample_index(rng: random.Random, probs: Sequence[float]) -> int:
    draw = rng.random()
    running = 0.0
    for idx, prob in enumerate(probs):
        running += prob
        if draw <= running:
            return idx
    return len(probs) - 1


def _estimate_beta(
    *,
    raw_samples: Sequence[float],
    reference_samples: Sequence[float],
) -> float:
    if len(raw_samples) <= 1:
        return 0.0
    raw_mean = float(mean(raw_samples))
    reference_mean = float(mean(reference_samples))
    centered_raw = [sample - raw_mean for sample in raw_samples]
    centered_reference = [sample - reference_mean for sample in reference_samples]
    var_reference = sum(value * value for value in centered_reference) / (len(reference_samples) - 1)
    if var_reference <= 0.0:
        return 0.0
    covariance = sum(
        raw_value * reference_value
        for raw_value, reference_value in zip(centered_raw, centered_reference)
    ) / (len(raw_samples) - 1)
    return covariance / var_reference


def _ci_width(summary: Mapping[str, Any]) -> float:
    low, high = summary["ci95_utility_per_hand"]
    return float(high - low)


def _reduction_pct(raw: float, adjusted: float) -> float:
    if raw <= 0.0:
        return 0.0
    return (1.0 - (adjusted / raw)) * 100.0


def _serialize_aivat_config(config: AIVATConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "rollout_count_per_action": config.rollout_count_per_action,
        "max_actions_per_rollout": config.max_actions_per_rollout,
        "max_branching_for_correction": config.max_branching_for_correction,
        "include_opponent_decisions": config.include_opponent_decisions,
    }


def _validate_league_inputs(
    policies: Mapping[str, _ActionPolicy],
    config: LeagueEvaluationConfig,
) -> None:
    if len(policies) < 2:
        raise ValueError("at least two policies are required for league evaluation")
    _validate_common_inputs(
        num_hands_per_seat=config.num_hands_per_seat,
        confidence_z=config.confidence_z,
        max_actions_per_hand=config.max_actions_per_hand,
    )
    _validate_optional_baseline(config.control_variate_baseline)
    _validate_aivat_config(config.aivat_config)


def _validate_proxy_inputs(
    policies: Mapping[str, BlueprintPolicy],
    config: ExploitabilityProxyConfig,
) -> None:
    if not policies:
        raise ValueError("at least one policy is required for exploitability proxy evaluation")
    _validate_common_inputs(
        num_hands_per_seat=config.num_hands_per_seat,
        confidence_z=config.confidence_z,
        max_actions_per_hand=config.max_actions_per_hand,
    )
    if not config.baseline_policies:
        raise ValueError("baseline_policies must not be empty")
    for baseline in config.baseline_policies:
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(f"unsupported baseline strategy: {baseline}")
    _validate_optional_baseline(config.control_variate_baseline)
    _validate_aivat_config(config.aivat_config)


def _validate_common_inputs(
    *,
    num_hands_per_seat: int,
    confidence_z: float,
    max_actions_per_hand: int,
) -> None:
    if num_hands_per_seat <= 0:
        raise ValueError("num_hands_per_seat must be positive")
    if confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")
    if max_actions_per_hand <= 0:
        raise ValueError("max_actions_per_hand must be positive")


def _validate_optional_baseline(value: BaselineStrategy | None) -> None:
    if value is None:
        return
    if value not in SUPPORTED_BASELINES:
        raise ValueError(f"unsupported baseline strategy: {value}")


def _validate_aivat_config(config: AIVATConfig | None) -> None:
    if config is None:
        return
    if config.rollout_count_per_action <= 0:
        raise ValueError("aivat rollout_count_per_action must be positive")
    if config.max_actions_per_rollout <= 0:
        raise ValueError("aivat max_actions_per_rollout must be positive")
    if config.max_branching_for_correction <= 0:
        raise ValueError("aivat max_branching_for_correction must be positive")
