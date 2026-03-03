from array import array
from typing import Iterable


INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
DEFAULT_REGRET_FLOOR = -310_000_000


def _new_int32_array(size: int) -> array:
    values = array("i", [0]) * size
    if values.itemsize != 4:
        raise RuntimeError("array('i') is not 4 bytes on this platform")
    return values


class LazyIntRegretTable:
    """
    Lazily allocated regret/strategy storage.

    Regrets are stored in 4-byte signed integers with floor clipping as required
    by the PRD notes from the Pluribus supplement.
    """

    def __init__(self, regret_floor: int = DEFAULT_REGRET_FLOOR) -> None:
        if regret_floor < INT32_MIN:
            raise ValueError("regret_floor must fit in int32")

        self.regret_floor = regret_floor
        self.regret_ceiling = INT32_MAX

        self._regrets: dict[str, array] = {}
        self._average_strategy: dict[str, list[float]] = {}
        self._allocations = 0

    @property
    def infoset_count(self) -> int:
        return len(self._regrets)

    @property
    def allocation_count(self) -> int:
        return self._allocations

    def keys(self) -> list[str]:
        return list(self._regrets.keys())

    def num_actions(self, key: str) -> int:
        if key not in self._regrets:
            raise KeyError(key)
        return len(self._regrets[key])

    def ensure_infoset(self, key: str, num_actions: int) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        existing = self._regrets.get(key)
        if existing is None:
            self._regrets[key] = _new_int32_array(num_actions)
            self._average_strategy[key] = [0.0] * num_actions
            self._allocations += 1
            return

        if len(existing) != num_actions:
            raise ValueError(
                f"infoset {key} has {len(existing)} actions but {num_actions} requested"
            )

    def get_regrets(self, key: str, num_actions: int) -> list[int]:
        self.ensure_infoset(key, num_actions)
        return [int(v) for v in self._regrets[key]]

    def get_average_strategy_sums(self, key: str, num_actions: int) -> list[float]:
        self.ensure_infoset(key, num_actions)
        return list(self._average_strategy[key])

    def current_strategy(self, key: str, num_actions: int) -> list[float]:
        regrets = self.get_regrets(key, num_actions)
        positive = [max(0.0, float(v)) for v in regrets]
        normalizer = sum(positive)
        if normalizer <= 0.0:
            uniform = 1.0 / num_actions
            return [uniform] * num_actions
        return [value / normalizer for value in positive]

    def average_strategy(self, key: str, num_actions: int) -> list[float]:
        sums = self.get_average_strategy_sums(key, num_actions)
        total = sum(sums)
        if total <= 0.0:
            uniform = 1.0 / num_actions
            return [uniform] * num_actions
        return [value / total for value in sums]

    def add_regret(self, key: str, action_index: int, delta: float, num_actions: int) -> None:
        self.ensure_infoset(key, num_actions)
        regrets = self._regrets[key]
        if action_index < 0 or action_index >= len(regrets):
            raise IndexError("action index out of range")

        updated = int(round(regrets[action_index] + delta))
        if updated < self.regret_floor:
            updated = self.regret_floor
        elif updated > self.regret_ceiling:
            updated = self.regret_ceiling
        regrets[action_index] = updated

    def add_regret_vector(self, key: str, deltas: Iterable[float], num_actions: int) -> None:
        self.ensure_infoset(key, num_actions)
        delta_values = list(deltas)
        if len(delta_values) != num_actions:
            raise ValueError("delta vector length mismatch")
        for i, delta in enumerate(delta_values):
            self.add_regret(key=key, action_index=i, delta=delta, num_actions=num_actions)

    def accumulate_average_strategy(
        self,
        key: str,
        strategy: Iterable[float],
        num_actions: int,
        weight: float,
    ) -> None:
        self.ensure_infoset(key, num_actions)
        values = list(strategy)
        if len(values) != num_actions:
            raise ValueError("strategy length mismatch")
        if weight < 0:
            raise ValueError("weight must be non-negative")

        target = self._average_strategy[key]
        for i, prob in enumerate(values):
            target[i] += weight * float(prob)

    def scale_all_regrets(self, factor: float) -> None:
        if factor < 0:
            raise ValueError("scale factor must be non-negative")
        for key, regrets in self._regrets.items():
            for i, value in enumerate(regrets):
                scaled = int(round(value * factor))
                if scaled < self.regret_floor:
                    scaled = self.regret_floor
                elif scaled > self.regret_ceiling:
                    scaled = self.regret_ceiling
                regrets[i] = scaled

    def scale_all_average_strategies(self, factor: float) -> None:
        if factor < 0:
            raise ValueError("scale factor must be non-negative")
        for key, values in self._average_strategy.items():
            for i, value in enumerate(values):
                values[i] = value * factor

    def snapshot_regrets(self) -> dict[str, list[int]]:
        return {key: [int(v) for v in regrets] for key, regrets in self._regrets.items()}

    def snapshot_average_strategy(self) -> dict[str, list[float]]:
        return {key: list(values) for key, values in self._average_strategy.items()}

    def serialize(self) -> dict[str, object]:
        return {
            "regret_floor": self.regret_floor,
            "regret_ceiling": self.regret_ceiling,
            "regrets": self.snapshot_regrets(),
            "average_strategy_sums": self.snapshot_average_strategy(),
        }

    @classmethod
    def deserialize(cls, payload: dict[str, object]) -> "LazyIntRegretTable":
        regret_floor = int(payload.get("regret_floor", DEFAULT_REGRET_FLOOR))
        table = cls(regret_floor=regret_floor)

        raw_regrets = payload.get("regrets", {})
        raw_avg = payload.get("average_strategy_sums", {})
        if not isinstance(raw_regrets, dict) or not isinstance(raw_avg, dict):
            raise ValueError("invalid serialized regret table payload")

        for key, regret_values in raw_regrets.items():
            if not isinstance(key, str):
                raise ValueError("infoset key must be a string")
            if not isinstance(regret_values, list) or not regret_values:
                raise ValueError("regret entry must be a non-empty list")

            num_actions = len(regret_values)
            table.ensure_infoset(key, num_actions)

            regrets = table._regrets[key]
            for i, value in enumerate(regret_values):
                regrets[i] = int(value)

            avg_values = raw_avg.get(key)
            if not isinstance(avg_values, list) or len(avg_values) != num_actions:
                raise ValueError("average strategy entry missing or length mismatch")
            table._average_strategy[key] = [float(v) for v in avg_values]

        return table
