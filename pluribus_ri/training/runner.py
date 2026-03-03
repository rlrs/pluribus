from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pluribus_ri.blueprint import BlueprintPolicy, run_blueprint_self_play
from pluribus_ri.solver import (
    ExternalSamplingLinearMCCFR,
    LazyIntRegretTable,
    MCCFRConfig,
    NLTHAbstractGameFactory,
    NLTHActionAbstractionConfig,
    NLTHGameConfig,
)

from .persistence import TrainingArtifactManager
from .snapshots import StrategySnapshot, build_strategy_snapshot


@dataclass(frozen=True)
class Phase1RunConfig:
    output_dir: str = "artifacts/phase1"
    iterations: int = 100
    checkpoint_interval: int = 25
    snapshot_interval: int = 25

    random_seed: int = 0

    # Core game config
    num_players: int = 6
    small_blind: int = 50
    big_blind: int = 100
    starting_stack: int = 10_000
    button: int = 0
    history_scope: str = "street"

    # Action abstraction
    preflop_raise_multipliers: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0)
    postflop_pot_raise_fractions: tuple[float, ...] = (0.5, 1.0, 2.0)
    flop_pot_raise_fractions: tuple[float, ...] | None = None
    turn_pot_raise_fractions: tuple[float, ...] | None = None
    river_pot_raise_fractions: tuple[float, ...] | None = None
    include_all_in: bool = True
    max_raise_actions: int = 4
    preflop_bucket_policy: str = "legacy"
    postflop_bucket_policy: str = "legacy"

    # MCCFR
    linear_weighting: bool = True
    discount_interval: int = 0
    regret_discount_factor: float = 1.0
    average_strategy_discount_factor: float = 1.0
    prune_after_iteration: int = 0
    negative_regret_pruning_threshold: int = -300_000_000
    explore_all_actions_probability: float = 0.05


@dataclass(frozen=True)
class Phase2RunConfig(Phase1RunConfig):
    output_dir: str = "artifacts/phase2"
    self_play_hands: int = 24
    self_play_seed: int = 0


@dataclass(frozen=True)
class _TrainingCoreOutputs:
    summary: dict[str, Any]
    artifact_manager: TrainingArtifactManager
    game_config: NLTHGameConfig
    abstraction_config: NLTHActionAbstractionConfig
    final_snapshot: StrategySnapshot


def run_phase1_training(config: Phase1RunConfig) -> dict[str, Any]:
    return _run_training_core(config=config, phase_label="phase_1").summary


def run_phase2_training(config: Phase2RunConfig) -> dict[str, Any]:
    core_outputs = _run_training_core(config=config, phase_label="phase_2")

    policy = BlueprintPolicy(
        iteration=core_outputs.final_snapshot.iteration,
        preflop_average=core_outputs.final_snapshot.preflop_average,
        postflop_current=core_outputs.final_snapshot.postflop_current,
    )
    blueprint_path = core_outputs.artifact_manager.write_blueprint(
        iteration=policy.iteration,
        payload=policy.to_dict(),
    )

    self_play_factory = NLTHAbstractGameFactory(
        game_config=core_outputs.game_config,
        abstraction_config=core_outputs.abstraction_config,
    )
    self_play = run_blueprint_self_play(
        policy=policy,
        game_factory=self_play_factory,
        num_hands=config.self_play_hands,
        random_seed=config.self_play_seed,
    )

    summary = dict(core_outputs.summary)
    summary["blueprint_policy_path"] = str(blueprint_path)
    summary["self_play"] = self_play
    summary_path = core_outputs.artifact_manager.write_summary(summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _run_training_core(
    config: Phase1RunConfig,
    phase_label: str,
) -> _TrainingCoreOutputs:
    game_config = NLTHGameConfig(
        num_players=config.num_players,
        small_blind=config.small_blind,
        big_blind=config.big_blind,
        starting_stack=config.starting_stack,
        button=config.button,
        random_seed=config.random_seed,
        history_scope=config.history_scope,  # type: ignore[arg-type]
    )

    abstraction_config = NLTHActionAbstractionConfig(
        preflop_raise_multipliers=config.preflop_raise_multipliers,
        postflop_pot_raise_fractions=config.postflop_pot_raise_fractions,
        flop_pot_raise_fractions=config.flop_pot_raise_fractions,
        turn_pot_raise_fractions=config.turn_pot_raise_fractions,
        river_pot_raise_fractions=config.river_pot_raise_fractions,
        include_all_in=config.include_all_in,
        max_raise_actions=config.max_raise_actions,
        preflop_bucket_policy=config.preflop_bucket_policy,  # type: ignore[arg-type]
        postflop_bucket_policy=config.postflop_bucket_policy,  # type: ignore[arg-type]
    )

    trainer_config = MCCFRConfig(
        iterations=config.iterations,
        random_seed=config.random_seed,
        linear_weighting=config.linear_weighting,
        discount_interval=config.discount_interval,
        regret_discount_factor=config.regret_discount_factor,
        average_strategy_discount_factor=config.average_strategy_discount_factor,
        prune_after_iteration=config.prune_after_iteration,
        negative_regret_pruning_threshold=config.negative_regret_pruning_threshold,
        explore_all_actions_probability=config.explore_all_actions_probability,
    )

    artifact_manager = TrainingArtifactManager(output_dir=config.output_dir)

    game_factory = NLTHAbstractGameFactory(
        game_config=game_config,
        abstraction_config=abstraction_config,
    )
    regret_table = LazyIntRegretTable()
    trainer = ExternalSamplingLinearMCCFR(regret_table=regret_table, config=trainer_config)

    written_checkpoints: list[str] = []
    written_snapshots: list[str] = []

    def on_iteration_end(iteration: int, _stats: Any) -> None:
        if config.checkpoint_interval > 0 and iteration % config.checkpoint_interval == 0:
            checkpoint_path = artifact_manager.write_checkpoint(
                iteration=iteration,
                trainer_config=trainer_config,
                game_config=game_config,
                abstraction_config=abstraction_config,
                training_stats=trainer.stats,
                regret_table=regret_table,
            )
            written_checkpoints.append(str(checkpoint_path))

        if config.snapshot_interval > 0 and iteration % config.snapshot_interval == 0:
            snapshot = build_strategy_snapshot(table=regret_table, iteration=iteration)
            snapshot_path = artifact_manager.write_snapshot(snapshot)
            written_snapshots.append(str(snapshot_path))

    stats = trainer.train_steps(
        root_state_factory=game_factory.root_state_factory(),
        num_players=config.num_players,
        iterations=config.iterations,
        on_iteration_end=on_iteration_end,
    )

    # Always persist terminal artifacts even when intervals do not divide iterations.
    if not written_checkpoints or not written_checkpoints[-1].endswith(f"{stats.iterations_completed:06d}.json"):
        checkpoint_path = artifact_manager.write_checkpoint(
            iteration=stats.iterations_completed,
            trainer_config=trainer_config,
            game_config=game_config,
            abstraction_config=abstraction_config,
            training_stats=stats,
            regret_table=regret_table,
        )
        written_checkpoints.append(str(checkpoint_path))

    final_snapshot = build_strategy_snapshot(table=regret_table, iteration=stats.iterations_completed)

    if not written_snapshots or not written_snapshots[-1].endswith(f"{stats.iterations_completed:06d}.json"):
        snapshot_path = artifact_manager.write_snapshot(final_snapshot)
        written_snapshots.append(str(snapshot_path))

    summary = {
        "phase": phase_label,
        "iterations_requested": config.iterations,
        "iterations_completed": stats.iterations_completed,
        "traversals_completed": stats.traversals_completed,
        "nodes_touched": stats.nodes_touched,
        "infosets_allocated": regret_table.infoset_count,
        "output_dir": str(Path(config.output_dir)),
        "checkpoints": written_checkpoints,
        "snapshots": written_snapshots,
    }
    summary_path = artifact_manager.write_summary(summary)
    summary["summary_path"] = str(summary_path)

    return _TrainingCoreOutputs(
        summary=summary,
        artifact_manager=artifact_manager,
        game_config=game_config,
        abstraction_config=abstraction_config,
        final_snapshot=final_snapshot,
    )
