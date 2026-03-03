from dataclasses import dataclass
import time


@dataclass(frozen=True)
class SubgameStoppingRules:
    min_cfr_iterations: int = 8
    max_cfr_iterations: int = 64
    max_nodes_touched: int = 75_000
    max_wallclock_ms: int = 250
    leaf_max_depth: int = 12


class SubgameSearchStopController:
    def __init__(self, rules: SubgameStoppingRules) -> None:
        if rules.min_cfr_iterations <= 0:
            raise ValueError("min_cfr_iterations must be positive")
        if rules.max_cfr_iterations < rules.min_cfr_iterations:
            raise ValueError("max_cfr_iterations must be >= min_cfr_iterations")
        if rules.max_nodes_touched <= 0:
            raise ValueError("max_nodes_touched must be positive")
        if rules.max_wallclock_ms <= 0:
            raise ValueError("max_wallclock_ms must be positive")
        if rules.leaf_max_depth <= 0:
            raise ValueError("leaf_max_depth must be positive")

        self.rules = rules
        self._start_time = time.monotonic()

    def should_stop(self, iterations: int, nodes_touched: int) -> tuple[bool, str]:
        if iterations < self.rules.min_cfr_iterations:
            return (False, "min_iterations_not_reached")

        if iterations >= self.rules.max_cfr_iterations:
            return (True, "max_iterations")

        if nodes_touched >= self.rules.max_nodes_touched:
            return (True, "max_nodes")

        elapsed_ms = (time.monotonic() - self._start_time) * 1000.0
        if elapsed_ms >= self.rules.max_wallclock_ms:
            return (True, "max_wallclock")

        return (False, "continue")
