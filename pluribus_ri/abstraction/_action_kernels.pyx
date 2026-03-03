# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from . import _py_action_kernels


cdef inline int _clamp(int value, int low, int high):
    if value < low:
        return low
    if value > high:
        return high
    return value


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _postflop_pot_raise_fractions(object abstraction_config, str street_value):
    if street_value == "flop" and abstraction_config.flop_pot_raise_fractions is not None:
        return abstraction_config.flop_pot_raise_fractions
    if street_value == "turn" and abstraction_config.turn_pot_raise_fractions is not None:
        return abstraction_config.turn_pot_raise_fractions
    if street_value == "river" and abstraction_config.river_pot_raise_fractions is not None:
        return abstraction_config.river_pot_raise_fractions
    return abstraction_config.postflop_pot_raise_fractions


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list _raise_targets(object engine, object abstraction_config, object legal):
    cdef object min_raise_to = legal.min_raise_to
    cdef object max_raise_to = legal.max_raise_to
    cdef int min_to
    cdef int max_to
    cdef set targets
    cdef object multiplier
    cdef int raw
    cdef tuple fractions
    cdef int base_pot
    cdef object fraction
    cdef int raise_size
    cdef int raise_to
    cdef list ordered
    cdef int target
    cdef int max_raise_actions
    cdef str street_value

    if min_raise_to is None or max_raise_to is None:
        return []

    min_to = int(min_raise_to)
    max_to = int(max_raise_to)
    if max_to < min_to:
        return []

    targets = set()
    street_value = engine.street.value
    if street_value == "preflop":
        for multiplier in abstraction_config.preflop_raise_multipliers:
            raw = int(round(multiplier * engine.big_blind))
            targets.add(_clamp(raw, min_to, max_to))
    else:
        fractions = _postflop_pot_raise_fractions(abstraction_config, street_value)
        base_pot = int(engine.total_pot)
        if base_pot < engine.big_blind:
            base_pot = engine.big_blind
        for fraction in fractions:
            raise_size = int(round(fraction * base_pot))
            if raise_size < engine.last_full_raise_size:
                raise_size = engine.last_full_raise_size
            raise_to = engine.current_bet + raise_size
            targets.add(_clamp(raise_to, min_to, max_to))

    if abstraction_config.include_all_in:
        targets.add(max_to)

    ordered = sorted(target for target in targets if min_to <= target <= max_to and target > engine.current_bet)
    max_raise_actions = int(abstraction_config.max_raise_actions)
    if max_raise_actions <= 0:
        return ordered
    if len(ordered) <= max_raise_actions:
        return ordered
    return _py_action_kernels._downsample_sorted_targets(ordered, max_raise_actions)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list legal_action_specs(object engine, object abstraction_config):
    cdef int seat
    cdef object legal
    cdef list actions = []
    cdef int raise_to

    if engine.hand_complete or engine.to_act is None:
        return actions

    seat = int(engine.to_act)
    legal = engine.get_legal_actions(seat)

    if legal.can_fold:
        actions.append(("fold", 0))

    if legal.can_check:
        actions.append(("check", 0))
    elif legal.call_amount > 0:
        actions.append(("call", 0))

    if legal.min_raise_to is not None and legal.max_raise_to is not None:
        for raise_to in _raise_targets(engine, abstraction_config, legal):
            actions.append(("raise", raise_to))

    return actions
