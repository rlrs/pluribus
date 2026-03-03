from dataclasses import dataclass
from enum import Enum
import random
from typing import Any, Literal

import eval7


RANKS = "23456789TJQKA"
SUITS = "cdhs"

ActionKind = Literal["fold", "check", "call", "raise"]


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    amount: int = 0


@dataclass(frozen=True)
class LegalActions:
    can_fold: bool
    can_check: bool
    call_amount: int
    min_raise_to: int | None
    max_raise_to: int | None

    def as_dict(self) -> dict[str, int | bool | None]:
        return {
            "can_fold": self.can_fold,
            "can_check": self.can_check,
            "call_amount": self.call_amount,
            "min_raise_to": self.min_raise_to,
            "max_raise_to": self.max_raise_to,
        }


@dataclass
class PlayerState:
    seat: int
    stack: int
    hole_cards: tuple[str, str] | None = None
    folded: bool = False
    all_in: bool = False
    contributed_street: int = 0
    contributed_total: int = 0

    @property
    def in_hand(self) -> bool:
        return not self.folded and self.hole_cards is not None


@dataclass(frozen=True)
class HandHistoryAction:
    seat: int
    street: str
    kind: ActionKind
    amount: int
    pot_after: int


class NoLimitHoldemEngine:
    """
    Deterministic six-player no-limit hold'em engine.

    The engine tracks full betting state, legal actions, side-pot showdown,
    and serializable hand histories for deterministic replay.
    """

    def __init__(
        self,
        num_players: int = 6,
        small_blind: int = 50,
        big_blind: int = 100,
        starting_stack: int = 10_000,
        seed: int | None = None,
    ) -> None:
        if num_players != 6:
            raise ValueError("v1 supports exactly six players")
        if small_blind <= 0 or big_blind <= 0:
            raise ValueError("blinds must be positive")
        if small_blind >= big_blind:
            raise ValueError("small blind must be lower than big blind")

        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed
        self.rng = random.Random(seed)

        self.starting_stack = starting_stack
        self.stacks: list[int] = [starting_stack for _ in range(num_players)]

        self.button = 0

        self.players: list[PlayerState] = []
        self.board: list[str] = []
        self.street = Street.COMPLETE
        self.current_bet = 0
        self.last_full_raise_size = self.big_blind
        self.to_act: int | None = None
        self.pending_to_act: set[int] = set()

        self.deck: list[str] = []
        self._deck_index = 0

        self.hand_complete = True
        self.winners: list[int] = []
        self.action_log: list[HandHistoryAction] = []

        self._hand_starting_stacks: list[int] = []
        self._blind_positions: tuple[int, int] = (0, 1)

    @staticmethod
    def full_deck() -> list[str]:
        return [rank + suit for rank in RANKS for suit in SUITS]

    def reset_match(self, stacks: list[int] | None = None, button: int = 0) -> None:
        if stacks is None:
            self.stacks = [self.starting_stack for _ in range(self.num_players)]
        else:
            if len(stacks) != self.num_players:
                raise ValueError("stacks length must equal num_players")
            self.stacks = [int(max(0, s)) for s in stacks]
        self.button = button % self.num_players
        self.hand_complete = True
        self.street = Street.COMPLETE

    def start_hand(self, deck_cards: list[str] | None = None) -> None:
        active_count = sum(1 for stack in self.stacks if stack > 0)
        if active_count < 2:
            raise RuntimeError("need at least two players with chips to start a hand")

        self.players = [PlayerState(seat=i, stack=self.stacks[i]) for i in range(self.num_players)]
        self._hand_starting_stacks = self.stacks[:]

        self.board = []
        self.street = Street.PREFLOP
        self.current_bet = 0
        self.last_full_raise_size = self.big_blind
        self.to_act = None
        self.pending_to_act = set()
        self.winners = []
        self.action_log = []
        self.hand_complete = False

        self.deck = self._build_deck(deck_cards)
        self._deck_index = 0

        self._deal_hole_cards()

        sb = self._next_seat_with_stack((self.button + 1) % self.num_players)
        bb = self._next_seat_with_stack((sb + 1) % self.num_players, exclude=sb)
        self._blind_positions = (sb, bb)

        self._post_blind(sb, self.small_blind)
        self._post_blind(bb, self.big_blind)

        self.current_bet = max(p.contributed_street for p in self.players)
        self.last_full_raise_size = self.big_blind

        self.pending_to_act = {seat for seat in range(self.num_players) if self._is_eligible_to_act(seat)}
        self.to_act = self._next_pending_from(bb)

        if self.to_act is None:
            self._runout_and_showdown()

    @property
    def total_pot(self) -> int:
        return sum(player.contributed_total for player in self.players)

    def clone(self, include_rng_state: bool = True) -> "NoLimitHoldemEngine":
        """
        Fast structural copy used by search traversals.

        This avoids generic ``copy.deepcopy`` recursion overhead while preserving
        engine state semantics.
        """

        cloned = self.__class__.__new__(self.__class__)

        cloned.num_players = self.num_players
        cloned.small_blind = self.small_blind
        cloned.big_blind = self.big_blind
        cloned.seed = self.seed
        cloned.starting_stack = self.starting_stack
        cloned.button = self.button

        if include_rng_state:
            cloned.rng = random.Random(0)
            cloned.rng.setstate(self.rng.getstate())
        else:
            # Runtime-search simulations never call deck-shuffling methods.
            # Reuse RNG reference to avoid expensive RNG reinitialization.
            cloned.rng = self.rng

        cloned.stacks = self.stacks[:]
        cloned.players = [
            PlayerState(
                seat=player.seat,
                stack=player.stack,
                hole_cards=player.hole_cards,
                folded=player.folded,
                all_in=player.all_in,
                contributed_street=player.contributed_street,
                contributed_total=player.contributed_total,
            )
            for player in self.players
        ]
        cloned.board = self.board[:]
        cloned.street = self.street
        cloned.current_bet = self.current_bet
        cloned.last_full_raise_size = self.last_full_raise_size
        cloned.to_act = self.to_act
        cloned.pending_to_act = set(self.pending_to_act)

        cloned.deck = self.deck[:]
        cloned._deck_index = self._deck_index

        cloned.hand_complete = self.hand_complete
        cloned.winners = self.winners[:]
        cloned.action_log = self.action_log[:]

        cloned._hand_starting_stacks = self._hand_starting_stacks[:]
        cloned._blind_positions = self._blind_positions
        return cloned

    def clone_for_simulation(self) -> "NoLimitHoldemEngine":
        return self.clone(include_rng_state=False)

    def __deepcopy__(self, memo: dict[int, object]) -> "NoLimitHoldemEngine":
        existing = memo.get(id(self))
        if existing is not None:
            return existing  # type: ignore[return-value]
        cloned = self.clone(include_rng_state=True)
        memo[id(self)] = cloned
        return cloned

    def get_legal_actions(self, seat: int | None = None) -> LegalActions:
        if self.hand_complete:
            raise RuntimeError("hand is complete")

        acting_seat = self.to_act if seat is None else seat
        if acting_seat is None:
            raise RuntimeError("no seat to act")
        if self.to_act != acting_seat:
            raise ValueError(f"seat {acting_seat} is not next to act")

        player = self.players[acting_seat]
        if not self._is_eligible_to_act(acting_seat):
            raise ValueError(f"seat {acting_seat} is not eligible to act")

        to_call = max(0, self.current_bet - player.contributed_street)
        call_amount = min(to_call, player.stack)

        can_check = to_call == 0
        can_fold = to_call > 0

        min_raise_to: int | None = None
        max_raise_to: int | None = None
        if player.stack > to_call:
            max_raise_to = player.contributed_street + player.stack
            full_raise_target = self.current_bet + self.last_full_raise_size
            if full_raise_target <= max_raise_to:
                min_raise_to = full_raise_target
            elif max_raise_to > self.current_bet:
                # Short all-in raise is still legal even if it is below full raise size.
                min_raise_to = max_raise_to

        return LegalActions(
            can_fold=can_fold,
            can_check=can_check,
            call_amount=call_amount,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
        )

    def apply_action(self, action: Action) -> None:
        if self.hand_complete:
            raise RuntimeError("cannot act on completed hand")
        if self.to_act is None:
            raise RuntimeError("no seat to act")

        seat = self.to_act
        player = self.players[seat]
        legal = self.get_legal_actions(seat)
        to_call = max(0, self.current_bet - player.contributed_street)
        logged_amount = 0

        if action.kind == "fold":
            if not legal.can_fold:
                raise ValueError("fold is illegal in this state")
            player.folded = True
            self.pending_to_act.discard(seat)

        elif action.kind == "check":
            if not legal.can_check:
                raise ValueError("check is illegal when facing a bet")
            self.pending_to_act.discard(seat)

        elif action.kind == "call":
            if legal.call_amount <= 0:
                raise ValueError("call is illegal when no chips are required")
            self._commit_chips(seat, legal.call_amount)
            logged_amount = legal.call_amount
            self.pending_to_act.discard(seat)

        elif action.kind == "raise":
            if legal.min_raise_to is None or legal.max_raise_to is None:
                raise ValueError("raise is illegal in this state")

            raise_to = int(action.amount)
            if raise_to < legal.min_raise_to or raise_to > legal.max_raise_to:
                raise ValueError(f"raise_to {raise_to} outside [{legal.min_raise_to}, {legal.max_raise_to}]")

            additional = raise_to - player.contributed_street
            if additional <= to_call:
                raise ValueError("raise must exceed call amount")

            previous_bet = self.current_bet
            self._commit_chips(seat, additional)
            self.current_bet = raise_to
            logged_amount = raise_to

            raise_size = raise_to - previous_bet
            if raise_size >= self.last_full_raise_size:
                self.last_full_raise_size = raise_size

            self.pending_to_act = {
                s for s in range(self.num_players) if s != seat and self._is_eligible_to_act(s)
            }

        else:
            raise ValueError(f"unknown action kind: {action.kind}")

        self.action_log.append(
            HandHistoryAction(
                seat=seat,
                street=self.street.value,
                kind=action.kind,
                amount=int(logged_amount),
                pot_after=self.total_pot,
            )
        )

        self._advance_after_action(last_actor=seat)

    def export_hand_history(self) -> dict[str, Any]:
        return {
            "num_players": self.num_players,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "button": self.button,
            "starting_stacks": self._hand_starting_stacks[:],
            "deck_order": self.deck[:],
            "actions": [
                {
                    "seat": action.seat,
                    "street": action.street,
                    "kind": action.kind,
                    "amount": action.amount,
                    "pot_after": action.pot_after,
                }
                for action in self.action_log
            ],
            "board": self.board[:],
            "final_stacks": self.stacks[:],
            "winners": self.winners[:],
        }

    @classmethod
    def replay_hand(cls, history: dict[str, Any]) -> "NoLimitHoldemEngine":
        engine = cls(
            num_players=int(history["num_players"]),
            small_blind=int(history["small_blind"]),
            big_blind=int(history["big_blind"]),
            starting_stack=0,
        )
        engine.reset_match(stacks=[int(v) for v in history["starting_stacks"]], button=int(history["button"]))
        engine.start_hand(deck_cards=list(history["deck_order"]))

        for raw in history["actions"]:
            action = Action(kind=str(raw["kind"]), amount=int(raw.get("amount", 0)))
            if engine.to_act != int(raw["seat"]):
                raise ValueError("history replay desynced: unexpected seat to act")
            engine.apply_action(action)

        return engine

    def _build_deck(self, prefix_cards: list[str] | None) -> list[str]:
        full = self.full_deck()
        if not prefix_cards:
            deck = full[:]
            self.rng.shuffle(deck)
            return deck

        seen: set[str] = set()
        for card in prefix_cards:
            if card not in full:
                raise ValueError(f"invalid card in deck prefix: {card}")
            if card in seen:
                raise ValueError(f"duplicate card in deck prefix: {card}")
            seen.add(card)

        remaining = [card for card in full if card not in seen]
        self.rng.shuffle(remaining)
        return list(prefix_cards) + remaining

    def _draw_card(self) -> str:
        if self._deck_index >= len(self.deck):
            raise RuntimeError("deck exhausted")
        card = self.deck[self._deck_index]
        self._deck_index += 1
        return card

    def _deal_hole_cards(self) -> None:
        order = self._active_order_from((self.button + 1) % self.num_players)
        hole_map: dict[int, list[str]] = {seat: [] for seat in order}
        for _ in range(2):
            for seat in order:
                hole_map[seat].append(self._draw_card())
        for seat in range(self.num_players):
            cards = hole_map.get(seat)
            if cards:
                self.players[seat].hole_cards = (cards[0], cards[1])
            else:
                self.players[seat].folded = True
                self.players[seat].all_in = True

    def _post_blind(self, seat: int, blind: int) -> None:
        self._commit_chips(seat, min(blind, self.players[seat].stack))

    def _commit_chips(self, seat: int, amount: int) -> int:
        player = self.players[seat]
        if amount < 0:
            raise ValueError("amount must be non-negative")

        actual = min(amount, player.stack)
        player.stack -= actual
        player.contributed_street += actual
        player.contributed_total += actual
        if player.stack == 0:
            player.all_in = True
        return actual

    def _is_eligible_to_act(self, seat: int) -> bool:
        player = self.players[seat]
        return player.in_hand and not player.all_in and player.stack > 0

    def _live_players(self) -> list[int]:
        return [seat for seat in range(self.num_players) if self.players[seat].in_hand]

    def _all_live_players_all_in(self) -> bool:
        live = self._live_players()
        return bool(live) and all(self.players[seat].all_in for seat in live)

    def _next_pending_from(self, start_seat: int) -> int | None:
        for offset in range(1, self.num_players + 1):
            seat = (start_seat + offset) % self.num_players
            if seat in self.pending_to_act and self._is_eligible_to_act(seat):
                return seat
        return None

    def _next_seat_with_stack(self, start_seat: int, exclude: int | None = None) -> int:
        for offset in range(self.num_players):
            seat = (start_seat + offset) % self.num_players
            if seat == exclude:
                continue
            if self.stacks[seat] > 0:
                return seat
        raise RuntimeError("no seat with chips found")

    def _active_order_from(self, start_seat: int) -> list[int]:
        order: list[int] = []
        for offset in range(self.num_players):
            seat = (start_seat + offset) % self.num_players
            if self.stacks[seat] > 0:
                order.append(seat)
        return order

    def _advance_after_action(self, last_actor: int) -> None:
        live = self._live_players()
        if len(live) == 1:
            self._award_pot_without_showdown(live[0])
            return

        if self._all_live_players_all_in():
            self._runout_and_showdown()
            return

        self.pending_to_act = {seat for seat in self.pending_to_act if self._is_eligible_to_act(seat)}

        if not self.pending_to_act:
            self._advance_street()
            return

        self.to_act = self._next_pending_from(last_actor)
        if self.to_act is None:
            self._advance_street()

    def _advance_street(self) -> None:
        if self.street == Street.PREFLOP:
            self.street = Street.FLOP
            self._deal_board_cards(3)
        elif self.street == Street.FLOP:
            self.street = Street.TURN
            self._deal_board_cards(1)
        elif self.street == Street.TURN:
            self.street = Street.RIVER
            self._deal_board_cards(1)
        elif self.street == Street.RIVER:
            self.street = Street.SHOWDOWN
            self._resolve_showdown()
            return
        else:
            raise RuntimeError(f"cannot advance street from {self.street}")

        for player in self.players:
            player.contributed_street = 0

        self.current_bet = 0
        self.last_full_raise_size = self.big_blind
        self.pending_to_act = {seat for seat in range(self.num_players) if self._is_eligible_to_act(seat)}

        if len(self._live_players()) == 1:
            self._award_pot_without_showdown(self._live_players()[0])
            return

        if self._all_live_players_all_in() or not self.pending_to_act:
            self._runout_and_showdown()
            return

        next_to_act = self._next_pending_from(self.button)
        if next_to_act is None:
            self._runout_and_showdown()
            return
        self.to_act = next_to_act

    def _deal_board_cards(self, n_cards: int) -> None:
        for _ in range(n_cards):
            self.board.append(self._draw_card())

    def _runout_and_showdown(self) -> None:
        while len(self.board) < 5:
            self._deal_board_cards(1)
        self.street = Street.SHOWDOWN
        self._resolve_showdown()

    def _award_pot_without_showdown(self, winner: int) -> None:
        self.players[winner].stack += self.total_pot
        self.winners = [winner]
        self._complete_hand()

    def _resolve_showdown(self) -> None:
        scores: dict[int, int] = {}
        for seat in self._live_players():
            player = self.players[seat]
            cards = [eval7.Card(card) for card in (*player.hole_cards, *self.board)]  # type: ignore[arg-type]
            scores[seat] = eval7.evaluate(cards)

        payouts = [0 for _ in range(self.num_players)]

        for pot_amount, eligible in self._build_side_pots():
            if not eligible:
                continue
            best_score = max(scores[seat] for seat in eligible)
            winners = [seat for seat in eligible if scores[seat] == best_score]

            base_share = pot_amount // len(winners)
            odd_chips = pot_amount % len(winners)

            for seat in winners:
                payouts[seat] += base_share

            for seat in self._odd_chip_order():
                if odd_chips == 0:
                    break
                if seat in winners:
                    payouts[seat] += 1
                    odd_chips -= 1

        for seat, amount in enumerate(payouts):
            self.players[seat].stack += amount

        max_payout = max(payouts) if payouts else 0
        self.winners = [seat for seat, amount in enumerate(payouts) if amount == max_payout and amount > 0]

        self._complete_hand()

    def _build_side_pots(self) -> list[tuple[int, list[int]]]:
        levels = sorted({p.contributed_total for p in self.players if p.contributed_total > 0})
        pots: list[tuple[int, list[int]]] = []

        prev = 0
        for level in levels:
            contributors = [seat for seat, player in enumerate(self.players) if player.contributed_total >= level]
            amount = (level - prev) * len(contributors)
            eligible = [seat for seat in contributors if self.players[seat].in_hand]
            if amount > 0:
                pots.append((amount, eligible))
            prev = level

        return pots

    def _odd_chip_order(self) -> list[int]:
        start = (self.button + 1) % self.num_players
        return [(start + offset) % self.num_players for offset in range(self.num_players)]

    def _complete_hand(self) -> None:
        self.hand_complete = True
        self.street = Street.COMPLETE
        self.to_act = None
        self.pending_to_act = set()
        self.stacks = [player.stack for player in self.players]
