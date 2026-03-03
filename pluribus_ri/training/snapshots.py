from dataclasses import dataclass

from pluribus_ri.solver.regret_table import LazyIntRegretTable


POSTFLOP_STREETS = {"flop", "turn", "river"}


@dataclass(frozen=True)
class StrategySnapshot:
    iteration: int
    preflop_average: dict[str, list[float]]
    postflop_current: dict[str, list[float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "preflop_average": self.preflop_average,
            "postflop_current": self.postflop_current,
            "preflop_infosets": len(self.preflop_average),
            "postflop_infosets": len(self.postflop_current),
        }


def extract_street_from_infoset_key(key: str) -> str | None:
    marker = "|street="
    start = key.find(marker)
    if start < 0:
        return None

    rest = key[start + len(marker) :]
    end = rest.find("|")
    return rest if end < 0 else rest[:end]


def build_strategy_snapshot(
    table: LazyIntRegretTable,
    iteration: int,
) -> StrategySnapshot:
    preflop_average: dict[str, list[float]] = {}
    postflop_current: dict[str, list[float]] = {}

    for key in table.keys():
        street = extract_street_from_infoset_key(key)
        if street is None:
            continue

        num_actions = table.num_actions(key)

        if street == "preflop":
            preflop_average[key] = table.average_strategy(key, num_actions)
        elif street in POSTFLOP_STREETS:
            postflop_current[key] = table.current_strategy(key, num_actions)

    return StrategySnapshot(
        iteration=iteration,
        preflop_average=preflop_average,
        postflop_current=postflop_current,
    )
