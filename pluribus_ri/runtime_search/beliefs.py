from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

from pluribus_ri.core import Action, NoLimitHoldemEngine


HandCombo = tuple[str, str]


class ActionLikelihoodModel(Protocol):
    """
    Interface for action-conditioned hand-likelihood models.

    Phase-3 scaffold intentionally keeps the model pluggable so future nested
    resolving can swap in richer estimators.
    """

    def likelihood(
        self,
        seat: int,
        hand: HandCombo,
        action: Action,
        engine: NoLimitHoldemEngine,
    ) -> float:
        ...


@dataclass
class PlayerBelief:
    seat: int
    probs: dict[HandCombo, float]

    def normalize(self) -> None:
        total = sum(self.probs.values())
        if total <= 0.0:
            uniform = 1.0 / len(self.probs)
            for hand in self.probs:
                self.probs[hand] = uniform
            return
        for hand in self.probs:
            self.probs[hand] /= total

    def top_hands(self, k: int = 5) -> list[tuple[HandCombo, float]]:
        if k <= 0:
            return []
        return sorted(self.probs.items(), key=lambda item: item[1], reverse=True)[:k]


class UniformActionLikelihoodModel:
    """Default likelihood model used by the scaffold before policy-informed models."""

    def likelihood(
        self,
        seat: int,
        hand: HandCombo,
        action: Action,
        engine: NoLimitHoldemEngine,
    ) -> float:
        del seat, hand, action, engine
        return 1.0


class OutsideObserverBeliefState:
    """
    Outside-observer marginal private-hand beliefs for each player.

    This Phase-3 scaffold keeps independent per-seat marginals instead of a full
    joint distribution over all private hands, which is enough to wire public-root
    and Bayesian update plumbing while staying computationally lightweight.
    """

    def __init__(
        self,
        players: dict[int, PlayerBelief],
        public_cards: tuple[str, ...],
    ) -> None:
        self.players = players
        self.public_cards = public_cards

    @classmethod
    def from_engine_public_state(
        cls,
        engine: NoLimitHoldemEngine,
    ) -> "OutsideObserverBeliefState":
        public_cards = tuple(engine.board)
        combos = all_private_hand_combos(excluded_cards=public_cards)
        uniform = 1.0 / len(combos)

        players: dict[int, PlayerBelief] = {}
        for seat in range(engine.num_players):
            players[seat] = PlayerBelief(
                seat=seat,
                probs={hand: uniform for hand in combos},
            )

        return cls(players=players, public_cards=public_cards)

    def apply_public_cards(
        self,
        public_cards: tuple[str, ...],
    ) -> None:
        self.public_cards = public_cards
        blocked = set(public_cards)
        for belief in self.players.values():
            belief.probs = {
                hand: prob
                for hand, prob in belief.probs.items()
                if hand[0] not in blocked and hand[1] not in blocked
            }
            belief.normalize()

    def observe_action(
        self,
        seat: int,
        action: Action,
        engine: NoLimitHoldemEngine,
        likelihood_model: ActionLikelihoodModel | None = None,
    ) -> None:
        belief = self.players.get(seat)
        if belief is None:
            raise ValueError(f"unknown seat {seat}")

        model = likelihood_model or UniformActionLikelihoodModel()
        for hand, prior in list(belief.probs.items()):
            weight = float(model.likelihood(seat=seat, hand=hand, action=action, engine=engine))
            if weight < 0.0:
                raise ValueError("likelihood model returned negative weight")
            belief.probs[hand] = prior * weight
        belief.normalize()

    def condition_on_known_hole_cards(self, seat: int, cards: HandCombo) -> None:
        if seat not in self.players:
            raise ValueError(f"unknown seat {seat}")

        ordered = _canonical_hand(cards)
        if len(set(ordered)) != 2:
            raise ValueError("hole cards must be two distinct cards")

        belief = self.players[seat]
        if ordered not in belief.probs:
            raise ValueError("known hole cards are incompatible with current public cards")

        belief.probs = {hand: 1.0 if hand == ordered else 0.0 for hand in belief.probs}
        belief.normalize()


def all_private_hand_combos(excluded_cards: tuple[str, ...] = ()) -> tuple[HandCombo, ...]:
    blocked = set(excluded_cards)
    deck = [card for card in NoLimitHoldemEngine.full_deck() if card not in blocked]
    combos = [_canonical_hand(pair) for pair in combinations(deck, 2)]
    return tuple(combos)


def _canonical_hand(cards: tuple[str, str]) -> HandCombo:
    c1, c2 = cards
    return (c1, c2) if c1 < c2 else (c2, c1)
