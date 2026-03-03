# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython


cdef inline bint _is_in_hand(object player):
    return (not player.folded) and (player.hole_cards is not None)


cdef inline bint _is_eligible(object player):
    return _is_in_hand(player) and (not player.all_in) and (player.stack > 0)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple live_player_count_and_last(object players, int num_players):
    cdef int count = 0
    cdef int last = -1
    cdef int seat
    cdef object player
    for seat in range(num_players):
        player = players[seat]
        if _is_in_hand(player):
            count += 1
            last = seat
    return count, last


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint all_live_players_all_in(object players, int num_players):
    cdef bint found_live = False
    cdef int seat
    cdef object player
    for seat in range(num_players):
        player = players[seat]
        if _is_in_hand(player):
            found_live = True
            if not player.all_in:
                return False
    return found_live


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef set eligible_seats(object players, int num_players, int exclude_seat=-1):
    cdef set out = set()
    cdef int seat
    cdef object player
    for seat in range(num_players):
        if seat == exclude_seat:
            continue
        player = players[seat]
        if _is_eligible(player):
            out.add(seat)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef set filter_pending_eligible(
    object pending_to_act,
    object players,
    int num_players,
):
    cdef set out = set()
    cdef int seat
    cdef object player

    for seat in pending_to_act:
        if seat < 0 or seat >= num_players:
            continue
        player = players[seat]
        if _is_eligible(player):
            out.add(seat)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object next_pending_from(
    int start_seat,
    object pending_to_act,
    object players,
    int num_players,
):
    cdef int offset
    cdef int seat
    cdef object player

    for offset in range(1, num_players + 1):
        seat = (start_seat + offset) % num_players
        if seat not in pending_to_act:
            continue
        player = players[seat]
        if _is_eligible(player):
            return seat
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int total_pot(object players, int num_players):
    cdef int out = 0
    cdef int seat
    cdef object player
    for seat in range(num_players):
        player = players[seat]
        out += int(player.contributed_total)
    return out
