from dataclasses import dataclass
import random
from typing import Callable, Literal, Sequence

from pluribus_ri.abstraction import (
    ActionAbstractionConfig,
    HistoryScope,
    NLTHAbstractGameBuilder,
)
from pluribus_ri.core import Action, NoLimitHoldemEngine

from .linear_mccfr import ExtensiveGameState


AbstractActionKind = Literal["fold", "check", "call", "raise"]


@dataclass(frozen=True)
class NLTHAbstractAction:
    kind: AbstractActionKind
    amount: int = 0

    def to_engine_action(self) -> Action:
        return Action(kind=self.kind, amount=self.amount)

    def token(self) -> str:
        if self.kind == "raise":
            return f"raise_to_{self.amount}"
        return self.kind


NLTHActionAbstractionConfig = ActionAbstractionConfig


@dataclass(frozen=True)
class NLTHGameConfig:
    num_players: int = 6
    small_blind: int = 50
    big_blind: int = 100
    starting_stack: int = 10_000
    button: int = 0
    random_seed: int | None = 0
    history_scope: HistoryScope = "street"


class NLTHAbstractGameState(ExtensiveGameState[NLTHAbstractAction]):
    """
    Adapter that makes the NLTH engine consumable by ExternalSamplingLinearMCCFR.

    Chance is resolved at hand start by dealing from a deterministic deck, so this
    state has only decision/terminal nodes.
    """

    def __init__(
        self,
        engine: NoLimitHoldemEngine,
        root_stacks: tuple[int, ...],
        abstraction_builder: NLTHAbstractGameBuilder,
    ) -> None:
        self.engine = engine
        self.root_stacks = root_stacks
        self.builder = abstraction_builder

    def is_terminal(self) -> bool:
        return self.engine.hand_complete

    def utility(self, player: int) -> float:
        if not self.is_terminal():
            raise ValueError("utility requested for non-terminal state")
        if player < 0 or player >= self.engine.num_players:
            raise ValueError("player out of range")
        return float(self.engine.stacks[player] - self.root_stacks[player])

    def is_chance_node(self) -> bool:
        return False

    def chance_outcomes(self) -> Sequence[tuple[NLTHAbstractAction, float]]:
        return []

    def current_player(self) -> int:
        if self.is_terminal() or self.engine.to_act is None:
            raise ValueError("current player requested for non-decision state")
        return int(self.engine.to_act)

    def legal_actions(self) -> Sequence[NLTHAbstractAction]:
        return [
            NLTHAbstractAction(kind=kind, amount=amount)
            for kind, amount in self.builder.legal_action_specs(self.engine)
        ]

    def child(self, action: NLTHAbstractAction) -> "NLTHAbstractGameState":
        if self.is_terminal():
            raise ValueError("cannot transition from terminal state")

        next_engine = self.engine.clone_for_simulation()
        next_engine.apply_action(action.to_engine_action())

        return NLTHAbstractGameState(
            engine=next_engine,
            root_stacks=self.root_stacks,
            abstraction_builder=self.builder,
        )

    def infoset_key(self, player: int) -> str:
        if self.is_terminal():
            raise ValueError("infoset key requested for terminal state")
        return self.builder.infoset_key(engine=self.engine, seat=player)

    def public_state_token(self) -> str:
        return self.builder.public_state_token(engine=self.engine)


class NLTHAbstractGameFactory:
    """Factory for deterministic root states usable by MCCFR traversals."""

    def __init__(
        self,
        game_config: NLTHGameConfig | None = None,
        abstraction_config: NLTHActionAbstractionConfig | None = None,
    ) -> None:
        self.game_config = game_config or NLTHGameConfig()
        self.abstraction_config = abstraction_config or NLTHActionAbstractionConfig()
        self.builder = NLTHAbstractGameBuilder(
            abstraction_config=self.abstraction_config,
            history_scope=self.game_config.history_scope,
        )

        self._rng = random.Random(self.game_config.random_seed)
        self._hand_counter = 0

    def root_state(self, deck_cards: list[str] | None = None) -> NLTHAbstractGameState:
        if deck_cards is None:
            hand_seed = self._rng.randint(0, 2**31 - 1)
        else:
            hand_seed = self.game_config.random_seed

        engine = NoLimitHoldemEngine(
            num_players=self.game_config.num_players,
            small_blind=self.game_config.small_blind,
            big_blind=self.game_config.big_blind,
            starting_stack=self.game_config.starting_stack,
            seed=hand_seed,
        )

        button = (self.game_config.button + self._hand_counter) % self.game_config.num_players
        engine.reset_match(
            stacks=[self.game_config.starting_stack for _ in range(self.game_config.num_players)],
            button=button,
        )
        engine.start_hand(deck_cards=deck_cards)

        self._hand_counter += 1

        return NLTHAbstractGameState(
            engine=engine,
            root_stacks=tuple(engine._hand_starting_stacks),
            abstraction_builder=self.builder,
        )

    def root_state_factory(self) -> Callable[[], NLTHAbstractGameState]:
        return self.root_state
