from typing import Literal

from pluribus_ri.core import NoLimitHoldemEngine, Street


AbstractActionKind = Literal["fold", "check", "call", "raise"]
AbstractActionSpec = tuple[AbstractActionKind, int]


def legal_action_specs(engine: NoLimitHoldemEngine, abstraction_config: object) -> list[AbstractActionSpec]:
    if engine.hand_complete or engine.to_act is None:
        return []

    seat = int(engine.to_act)
    legal = engine.get_legal_actions(seat)

    actions: list[AbstractActionSpec] = []
    if legal.can_fold:
        actions.append(("fold", 0))

    if legal.can_check:
        actions.append(("check", 0))
    elif legal.call_amount > 0:
        actions.append(("call", 0))

    if legal.min_raise_to is not None and legal.max_raise_to is not None:
        for raise_to in _raise_targets(engine=engine, abstraction_config=abstraction_config, legal=legal):
            actions.append(("raise", raise_to))

    return actions


def _raise_targets(engine: NoLimitHoldemEngine, abstraction_config: object, legal: object) -> list[int]:
    if legal.min_raise_to is None or legal.max_raise_to is None:
        return []

    min_to = legal.min_raise_to
    max_to = legal.max_raise_to
    if max_to < min_to:
        return []

    targets: set[int] = set()
    if engine.street == Street.PREFLOP:
        for multiplier in abstraction_config.preflop_raise_multipliers:
            raw = int(round(multiplier * engine.big_blind))
            targets.add(_clamp(raw, low=min_to, high=max_to))
    else:
        fractions = _postflop_pot_raise_fractions(abstraction_config=abstraction_config, street=engine.street)
        base_pot = max(engine.total_pot, engine.big_blind)
        for fraction in fractions:
            raise_size = int(round(fraction * base_pot))
            raise_to = engine.current_bet + max(raise_size, engine.last_full_raise_size)
            targets.add(_clamp(raise_to, low=min_to, high=max_to))

    if abstraction_config.include_all_in:
        targets.add(max_to)

    ordered = sorted(
        target
        for target in targets
        if min_to <= target <= max_to and target > engine.current_bet
    )

    if abstraction_config.max_raise_actions <= 0:
        return ordered
    if len(ordered) <= abstraction_config.max_raise_actions:
        return ordered
    return _downsample_sorted_targets(ordered, abstraction_config.max_raise_actions)


def _postflop_pot_raise_fractions(abstraction_config: object, street: Street) -> tuple[float, ...]:
    if street == Street.FLOP and abstraction_config.flop_pot_raise_fractions is not None:
        return abstraction_config.flop_pot_raise_fractions
    if street == Street.TURN and abstraction_config.turn_pot_raise_fractions is not None:
        return abstraction_config.turn_pot_raise_fractions
    if street == Street.RIVER and abstraction_config.river_pot_raise_fractions is not None:
        return abstraction_config.river_pot_raise_fractions
    return abstraction_config.postflop_pot_raise_fractions


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _downsample_sorted_targets(values: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]

    indices = [round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)]

    out: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        value = values[idx]
        if value in seen:
            continue
        seen.add(value)
        out.append(value)

    if len(out) < limit:
        for value in values:
            if value in seen:
                continue
            out.append(value)
            seen.add(value)
            if len(out) == limit:
                break
    return out
