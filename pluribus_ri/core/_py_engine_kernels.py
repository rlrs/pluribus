def _is_in_hand(player: object) -> bool:
    return (not player.folded) and (player.hole_cards is not None)


def _is_eligible(player: object) -> bool:
    return _is_in_hand(player) and (not player.all_in) and (player.stack > 0)


def live_player_count_and_last(players: list[object], num_players: int) -> tuple[int, int]:
    count = 0
    last = -1
    for seat in range(num_players):
        player = players[seat]
        if _is_in_hand(player):
            count += 1
            last = seat
    return count, last


def all_live_players_all_in(players: list[object], num_players: int) -> bool:
    found_live = False
    for seat in range(num_players):
        player = players[seat]
        if _is_in_hand(player):
            found_live = True
            if not player.all_in:
                return False
    return found_live


def eligible_seats(players: list[object], num_players: int, exclude_seat: int = -1) -> set[int]:
    out: set[int] = set()
    for seat in range(num_players):
        if seat == exclude_seat:
            continue
        player = players[seat]
        if _is_eligible(player):
            out.add(seat)
    return out


def filter_pending_eligible(
    pending_to_act: set[int],
    players: list[object],
    num_players: int,
) -> set[int]:
    out: set[int] = set()
    for seat in pending_to_act:
        if seat < 0 or seat >= num_players:
            continue
        if _is_eligible(players[seat]):
            out.add(seat)
    return out


def next_pending_from(
    start_seat: int,
    pending_to_act: set[int],
    players: list[object],
    num_players: int,
) -> int | None:
    for offset in range(1, num_players + 1):
        seat = (start_seat + offset) % num_players
        if seat not in pending_to_act:
            continue
        if _is_eligible(players[seat]):
            return seat
    return None


def total_pot(players: list[object], num_players: int) -> int:
    out = 0
    for seat in range(num_players):
        out += int(players[seat].contributed_total)
    return out
