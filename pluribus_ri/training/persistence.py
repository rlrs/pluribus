from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from pluribus_ri.solver.linear_mccfr import MCCFRConfig, TrainingStats
from pluribus_ri.solver.nlth_game import NLTHActionAbstractionConfig, NLTHGameConfig
from pluribus_ri.solver.regret_table import LazyIntRegretTable

from .snapshots import StrategySnapshot


class TrainingArtifactManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.snapshots_dir = self.output_dir / "snapshots"
        self.blueprints_dir = self.output_dir / "blueprints"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.blueprints_dir.mkdir(parents=True, exist_ok=True)

    def write_checkpoint(
        self,
        iteration: int,
        trainer_config: MCCFRConfig,
        game_config: NLTHGameConfig,
        abstraction_config: NLTHActionAbstractionConfig,
        training_stats: TrainingStats,
        regret_table: LazyIntRegretTable,
    ) -> Path:
        path = self.checkpoints_dir / f"checkpoint_iter_{iteration:06d}.json"
        payload = {
            "iteration": iteration,
            "trainer_config": _asdict_fallback(trainer_config),
            "game_config": _asdict_fallback(game_config),
            "abstraction_config": _asdict_fallback(abstraction_config),
            "training_stats": _asdict_fallback(training_stats),
            "regret_table": regret_table.serialize(),
        }
        _write_json(path, payload)
        return path

    def write_snapshot(self, snapshot: StrategySnapshot) -> Path:
        path = self.snapshots_dir / f"snapshot_iter_{snapshot.iteration:06d}.json"
        _write_json(path, snapshot.to_dict())
        return path

    def write_summary(self, payload: dict[str, Any]) -> Path:
        path = self.output_dir / "summary.json"
        _write_json(path, payload)
        return path

    def write_blueprint(self, iteration: int, payload: dict[str, Any]) -> Path:
        path = self.blueprints_dir / f"blueprint_iter_{iteration:06d}.json"
        _write_json(path, payload)
        return path


def _asdict_fallback(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"expected dataclass instance, got {type(value)}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
