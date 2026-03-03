from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .game_builder import ActionAbstractionConfig
from .state_indexer import HistoryScope


@dataclass(frozen=True)
class AbstractionTablesConfig:
    """
    Dedicated abstraction-table configuration used for Phase 4 tuning.

    The file-backed config controls both action abstraction (raise tables) and
    preflop private-hand bucket policy.
    """

    history_scope: HistoryScope = "street"
    action: ActionAbstractionConfig = field(default_factory=ActionAbstractionConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AbstractionTablesConfig":
        history_scope_raw = payload.get("history_scope", "street")
        history_scope = str(history_scope_raw)
        if history_scope not in {"street", "all"}:
            raise ValueError(f"unsupported history_scope: {history_scope}")

        action_raw = payload.get("action", {})
        if not isinstance(action_raw, dict):
            raise ValueError("action section must be a dictionary")

        preflop_bucket_policy_raw = action_raw.get("preflop_bucket_policy", "legacy")
        preflop_bucket_policy = str(preflop_bucket_policy_raw)
        if preflop_bucket_policy not in {"legacy", "canonical169"}:
            raise ValueError(f"unsupported preflop_bucket_policy: {preflop_bucket_policy}")
        postflop_bucket_policy_raw = action_raw.get("postflop_bucket_policy", "legacy")
        postflop_bucket_policy = str(postflop_bucket_policy_raw)
        if postflop_bucket_policy not in {"legacy", "texture_v1"}:
            raise ValueError(f"unsupported postflop_bucket_policy: {postflop_bucket_policy}")

        return cls(
            history_scope=history_scope,  # type: ignore[arg-type]
            action=ActionAbstractionConfig(
                preflop_raise_multipliers=_float_tuple(
                    action_raw.get("preflop_raise_multipliers", (2.0, 3.0, 5.0, 10.0))
                ),
                postflop_pot_raise_fractions=_float_tuple(
                    action_raw.get("postflop_pot_raise_fractions", (0.5, 1.0, 2.0))
                ),
                flop_pot_raise_fractions=_optional_float_tuple(action_raw.get("flop_pot_raise_fractions")),
                turn_pot_raise_fractions=_optional_float_tuple(action_raw.get("turn_pot_raise_fractions")),
                river_pot_raise_fractions=_optional_float_tuple(action_raw.get("river_pot_raise_fractions")),
                include_all_in=bool(action_raw.get("include_all_in", True)),
                max_raise_actions=int(action_raw.get("max_raise_actions", 4)),
                preflop_bucket_policy=preflop_bucket_policy,  # type: ignore[arg-type]
                postflop_bucket_policy=postflop_bucket_policy,  # type: ignore[arg-type]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_scope": self.history_scope,
            "action": {
                "preflop_raise_multipliers": list(self.action.preflop_raise_multipliers),
                "postflop_pot_raise_fractions": list(self.action.postflop_pot_raise_fractions),
                "flop_pot_raise_fractions": _tuple_or_none(self.action.flop_pot_raise_fractions),
                "turn_pot_raise_fractions": _tuple_or_none(self.action.turn_pot_raise_fractions),
                "river_pot_raise_fractions": _tuple_or_none(self.action.river_pot_raise_fractions),
                "include_all_in": self.action.include_all_in,
                "max_raise_actions": self.action.max_raise_actions,
                "preflop_bucket_policy": self.action.preflop_bucket_policy,
                "postflop_bucket_policy": self.action.postflop_bucket_policy,
            },
        }


def load_abstraction_tables_config(path: str | Path) -> AbstractionTablesConfig:
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("abstraction config root must be a dictionary")
    return AbstractionTablesConfig.from_dict(payload)


def write_abstraction_tables_config(config: AbstractionTablesConfig, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return file_path


def _float_tuple(raw: object) -> tuple[float, ...]:
    if isinstance(raw, tuple):
        values = [float(value) for value in raw]
    elif isinstance(raw, list):
        values = [float(value) for value in raw]
    else:
        raise ValueError("expected list/tuple of numeric values")
    if not values:
        raise ValueError("numeric tuple cannot be empty")
    return tuple(values)


def _optional_float_tuple(raw: object) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return _float_tuple(raw)


def _tuple_or_none(values: tuple[float, ...] | None) -> list[float] | None:
    if values is None:
        return None
    return list(values)
