from typing import Literal

import eval7

from pluribus_ri.core import NoLimitHoldemEngine

from .infoset import PublicStateKey, encode_infoset_key


HistoryScope = Literal["street", "all"]
PreflopBucketPolicy = Literal["legacy", "canonical169"]
PostflopBucketPolicy = Literal["legacy", "texture_v1"]


_RANK_TO_INT = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

_HAND_TYPE_TO_BUCKET = {
    "High Card": 0,
    "Pair": 1,
    "Two Pair": 2,
    "Trips": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Quads": 7,
    "Straight Flush": 8,
}


def build_public_state_key(
    engine: NoLimitHoldemEngine,
    history_scope: HistoryScope = "street",
) -> PublicStateKey:
    if engine.to_act is None and not engine.hand_complete:
        raise ValueError("engine has no active seat to act")

    street_name = engine.street.value

    if history_scope == "street":
        action_history = tuple(
            f"p{a.seat}:{a.kind}:{a.amount}"
            for a in engine.action_log
            if a.street == street_name
        )
    elif history_scope == "all":
        action_history = tuple(f"{a.street}:p{a.seat}:{a.kind}:{a.amount}" for a in engine.action_log)
    else:
        raise ValueError(f"unsupported history scope: {history_scope}")

    return PublicStateKey(
        street=street_name,
        board_cards=tuple(engine.board),
        to_act=-1 if engine.to_act is None else int(engine.to_act),
        pot=int(engine.total_pot),
        current_bet=int(engine.current_bet),
        stacks=tuple(int(player.stack) for player in engine.players),
        contributed_street=tuple(int(player.contributed_street) for player in engine.players),
        active_mask=tuple(1 if player.in_hand else 0 for player in engine.players),
        action_history=action_history,
    )


def private_hand_bucket(engine: NoLimitHoldemEngine, seat: int) -> int:
    return private_hand_bucket_with_policy(engine=engine, seat=seat, preflop_bucket_policy="legacy")


def private_hand_bucket_with_policy(
    engine: NoLimitHoldemEngine,
    seat: int,
    preflop_bucket_policy: PreflopBucketPolicy = "legacy",
    postflop_bucket_policy: PostflopBucketPolicy = "legacy",
) -> int:
    if seat < 0 or seat >= engine.num_players:
        raise ValueError("seat out of range")

    player = engine.players[seat]
    if player.hole_cards is None:
        return -1

    c1, c2 = player.hole_cards

    if len(engine.board) == 0:
        return _preflop_bucket(c1, c2, preflop_bucket_policy)

    return _postflop_bucket(c1, c2, tuple(engine.board), postflop_bucket_policy)


def encode_engine_infoset_key(
    engine: NoLimitHoldemEngine,
    seat: int,
    history_scope: HistoryScope = "street",
    preflop_bucket_policy: PreflopBucketPolicy = "legacy",
    postflop_bucket_policy: PostflopBucketPolicy = "legacy",
) -> str:
    public_key = build_public_state_key(engine=engine, history_scope=history_scope)
    bucket = private_hand_bucket_with_policy(
        engine=engine,
        seat=seat,
        preflop_bucket_policy=preflop_bucket_policy,
        postflop_bucket_policy=postflop_bucket_policy,
    )
    return encode_infoset_key(seat=seat, private_bucket=bucket, public_state=public_key)


def _preflop_bucket(
    card_a: str,
    card_b: str,
    policy: PreflopBucketPolicy,
) -> int:
    if policy == "legacy":
        return _preflop_bucket_legacy(card_a, card_b)
    if policy == "canonical169":
        return _preflop_bucket_canonical169(card_a, card_b)
    raise ValueError(f"unsupported preflop bucket policy: {policy}")


def _preflop_bucket_legacy(card_a: str, card_b: str) -> int:
    rank_a = _RANK_TO_INT[card_a[0]]
    rank_b = _RANK_TO_INT[card_b[0]]
    suited = int(card_a[1] == card_b[1])

    high = max(rank_a, rank_b)
    low = min(rank_a, rank_b)

    # Pairs occupy a dedicated deterministic range.
    if rank_a == rank_b:
        return 100 + high

    # Non-pair buckets are rank-ordered and suit-aware.
    return high * 32 + low * 2 + suited


def _preflop_bucket_canonical169(card_a: str, card_b: str) -> int:
    """
    Canonical 169 preflop abstraction.

    Buckets are partitioned into:
    - 13 pair buckets
    - 78 suited non-pair buckets
    - 78 offsuit non-pair buckets
    """

    rank_a = _RANK_TO_INT[card_a[0]] - 2
    rank_b = _RANK_TO_INT[card_b[0]] - 2

    if rank_a == rank_b:
        return rank_a

    high = max(rank_a, rank_b)
    low = min(rank_a, rank_b)
    suited = card_a[1] == card_b[1]

    nonpair_index = high * (high - 1) // 2 + low
    base = 13
    if suited:
        return base + nonpair_index
    return base + 78 + nonpair_index


def _postflop_bucket(
    card_a: str,
    card_b: str,
    board: tuple[str, ...],
    policy: PostflopBucketPolicy,
) -> int:
    if policy == "legacy":
        return _postflop_bucket_legacy(card_a, card_b, board)
    if policy == "texture_v1":
        return _postflop_bucket_texture_v1(card_a, card_b, board)
    raise ValueError(f"unsupported postflop bucket policy: {policy}")


def _postflop_bucket_legacy(card_a: str, card_b: str, board: tuple[str, ...]) -> int:
    cards = [eval7.Card(card) for card in (card_a, card_b, *board)]
    score = int(eval7.evaluate(cards))
    hand_type = eval7.handtype(score)

    coarse = _HAND_TYPE_TO_BUCKET.get(hand_type, 0)
    # Deterministic fine component to keep buckets informative without requiring
    # unpublished clustering artifacts from the original system.
    fine = score % 32
    return coarse * 32 + fine


def _postflop_bucket_texture_v1(card_a: str, card_b: str, board: tuple[str, ...]) -> int:
    """
    Feature bucket for postflop abstraction with explicit board/draw structure.

    This policy keeps determinism and low compute cost while improving semantic
    grouping over the legacy handtype+mod bucket.
    """

    cards = [eval7.Card(card) for card in (card_a, card_b, *board)]
    score = int(eval7.evaluate(cards))
    hand_type = eval7.handtype(score)
    coarse = _HAND_TYPE_TO_BUCKET.get(hand_type, 0)

    score_bin = min(63, score >> 21)
    draw_code = _draw_code(card_a, card_b, board, hand_type)
    board_code = _board_texture_code(board)
    hole_code = _hole_feature_code(card_a, card_b, board, hand_type)

    # Bit-packed composition:
    # coarse(4b) | score(6b) | draw(2b) | board(3b) | hole(3b)
    return (coarse << 14) | (score_bin << 8) | (draw_code << 6) | (board_code << 3) | hole_code


def _draw_code(card_a: str, card_b: str, board: tuple[str, ...], hand_type: str) -> int:
    if len(board) >= 5:
        return 0
    if hand_type in {"Straight", "Flush", "Full House", "Quads", "Straight Flush"}:
        return 0

    flush_draw = _has_flush_draw(card_a, card_b, board)
    straight_draw = _has_straight_draw(card_a, card_b, board)

    if flush_draw and straight_draw:
        return 3
    if flush_draw:
        return 2
    if straight_draw:
        return 1
    return 0


def _has_flush_draw(card_a: str, card_b: str, board: tuple[str, ...]) -> bool:
    suits = [card_a[1], card_b[1], *(card[1] for card in board)]
    counts: dict[str, int] = {}
    for suit in suits:
        counts[suit] = counts.get(suit, 0) + 1
    return max(counts.values(), default=0) >= 4


def _has_straight_draw(card_a: str, card_b: str, board: tuple[str, ...]) -> bool:
    ranks = {_RANK_TO_INT[card_a[0]], _RANK_TO_INT[card_b[0]], *(_RANK_TO_INT[card[0]] for card in board)}
    if 14 in ranks:
        ranks.add(1)

    for start in range(1, 11):
        run = {start + offset for offset in range(5)}
        present = len(run.intersection(ranks))
        if present == 4:
            return True
    return False


def _board_texture_code(board: tuple[str, ...]) -> int:
    suits = [card[1] for card in board]
    suit_types = len(set(suits))

    monotone = int(suit_types == 1)
    two_tone = int(suit_types == 2)

    ranks = [card[0] for card in board]
    paired = int(len(set(ranks)) < len(ranks))

    # 0..7
    return (paired << 2) | (monotone << 1) | two_tone


def _hole_feature_code(card_a: str, card_b: str, board: tuple[str, ...], hand_type: str) -> int:
    rank_a = _RANK_TO_INT[card_a[0]]
    rank_b = _RANK_TO_INT[card_b[0]]
    board_max = max(_RANK_TO_INT[card[0]] for card in board)

    overcards = int(rank_a > board_max) + int(rank_b > board_max)
    pocket_pair = int(rank_a == rank_b)
    top_pair = int(hand_type == "Pair" and (rank_a == board_max or rank_b == board_max))

    # 0..7
    return (pocket_pair << 2) | (top_pair << 1) | min(overcards, 1)
