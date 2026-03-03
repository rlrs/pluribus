from dataclasses import dataclass

from pluribus_ri.abstraction import HistoryScope, build_public_state_key
from pluribus_ri.core import Action, NoLimitHoldemEngine, Street

from .beliefs import OutsideObserverBeliefState


_STREET_ORDER: dict[str, int] = {
    Street.PREFLOP.value: 0,
    Street.FLOP.value: 1,
    Street.TURN.value: 2,
    Street.RIVER.value: 3,
}


@dataclass(frozen=True)
class PublicSearchRoot:
    """
    Runtime-search root rebuilt from the start of the current betting round.

    Phase-3 scaffold includes a round-start engine snapshot and outside-observer
    beliefs so nested re-solving components can consume a deterministic root.
    """

    street: str
    round_start_engine: NoLimitHoldemEngine
    current_engine: NoLimitHoldemEngine
    public_state_token: str
    beliefs: OutsideObserverBeliefState


def build_public_search_root(
    engine: NoLimitHoldemEngine,
    history_scope: HistoryScope = "street",
) -> PublicSearchRoot:
    if engine.hand_complete:
        raise ValueError("cannot build public search root from a completed hand")
    if engine.street not in {Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER}:
        raise ValueError(f"unsupported street for runtime root: {engine.street}")

    current_street = engine.street.value
    round_start_engine = _rebuild_engine_from_street_start(engine, current_street=current_street)
    beliefs = OutsideObserverBeliefState.from_engine_public_state(round_start_engine)

    return PublicSearchRoot(
        street=current_street,
        round_start_engine=round_start_engine,
        current_engine=engine,
        public_state_token=build_public_state_key(
            engine=round_start_engine,
            history_scope=history_scope,
        ).to_token(),
        beliefs=beliefs,
    )


def _rebuild_engine_from_street_start(
    engine: NoLimitHoldemEngine,
    current_street: str,
) -> NoLimitHoldemEngine:
    history = engine.export_hand_history()
    prefix_actions = [
        raw
        for raw in history["actions"]
        if _STREET_ORDER.get(str(raw["street"]), 99) < _STREET_ORDER[current_street]
    ]

    rebuilt = NoLimitHoldemEngine(
        num_players=int(history["num_players"]),
        small_blind=int(history["small_blind"]),
        big_blind=int(history["big_blind"]),
        starting_stack=0,
    )
    rebuilt.reset_match(
        stacks=[int(v) for v in history["starting_stacks"]],
        button=int(history["button"]),
    )
    rebuilt.start_hand(deck_cards=list(history["deck_order"]))

    for raw in prefix_actions:
        if rebuilt.to_act != int(raw["seat"]):
            raise RuntimeError("rebuild desynced while replaying prior-round actions")
        rebuilt.apply_action(Action(kind=str(raw["kind"]), amount=int(raw.get("amount", 0))))

    if rebuilt.street.value != current_street:
        raise RuntimeError(
            f"round-start rebuild ended on {rebuilt.street.value}, expected {current_street}"
        )

    return rebuilt
