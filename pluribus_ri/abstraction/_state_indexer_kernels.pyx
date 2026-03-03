# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef str build_public_state_token(object engine, str history_scope):
    cdef str street_name
    cdef object action
    cdef list history_parts = []
    cdef int seat
    cdef object player
    cdef list stacks_parts = []
    cdef list contrib_parts = []
    cdef list active_parts = []
    cdef int to_act
    cdef str action_history
    cdef str board
    cdef str stacks
    cdef str street_contrib
    cdef str active

    if engine.to_act is None and not engine.hand_complete:
        raise ValueError("engine has no active seat to act")

    street_name = engine.street.value
    if history_scope == "street":
        for action in engine.action_log:
            if action.street == street_name:
                history_parts.append(f"p{action.seat}:{action.kind}:{action.amount}")
    elif history_scope == "all":
        for action in engine.action_log:
            history_parts.append(f"{action.street}:p{action.seat}:{action.kind}:{action.amount}")
    else:
        raise ValueError(f"unsupported history scope: {history_scope}")

    board = "".join(engine.board)
    for seat in range(engine.num_players):
        player = engine.players[seat]
        stacks_parts.append(str(int(player.stack)))
        contrib_parts.append(str(int(player.contributed_street)))
        active_parts.append("1" if ((not player.folded) and (player.hole_cards is not None)) else "0")

    stacks = ",".join(stacks_parts)
    street_contrib = ",".join(contrib_parts)
    active = "".join(active_parts)
    action_history = ";".join(history_parts)
    to_act = -1 if engine.to_act is None else int(engine.to_act)

    return (
        f"street={street_name}|board={board}|to_act={to_act}|pot={int(engine.total_pot)}|bet={int(engine.current_bet)}|"
        f"stacks={stacks}|c={street_contrib}|active={active}|hist={action_history}"
    )
