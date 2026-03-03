from dataclasses import dataclass
from typing import Literal

from pluribus_ri.core import NoLimitHoldemEngine, Street

from .state_indexer import (
    HistoryScope,
    PostflopBucketPolicy,
    PreflopBucketPolicy,
    build_public_state_key,
    encode_engine_infoset_key,
)


AbstractActionKind = Literal["fold", "check", "call", "raise"]
AbstractActionSpec = tuple[AbstractActionKind, int]


@dataclass(frozen=True)
class ActionAbstractionConfig:
    """Explicit, configurable v1 abstraction choices for raise sizing."""

    preflop_raise_multipliers: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0)
    postflop_pot_raise_fractions: tuple[float, ...] = (0.5, 1.0, 2.0)
    flop_pot_raise_fractions: tuple[float, ...] | None = None
    turn_pot_raise_fractions: tuple[float, ...] | None = None
    river_pot_raise_fractions: tuple[float, ...] | None = None
    include_all_in: bool = True
    max_raise_actions: int = 4
    preflop_bucket_policy: PreflopBucketPolicy = "legacy"
    postflop_bucket_policy: PostflopBucketPolicy = "legacy"


class NLTHAbstractGameBuilder:
    """
    Dedicated abstraction-layer builder for blueprint game states.

    It centralizes action abstraction plus infoset/public-state keying so the
    solver adapter can stay thin and deterministic.
    """

    def __init__(
        self,
        abstraction_config: ActionAbstractionConfig | None = None,
        history_scope: HistoryScope = "street",
    ) -> None:
        self.abstraction_config = abstraction_config or ActionAbstractionConfig()
        self.history_scope = history_scope

    def legal_action_specs(self, engine: NoLimitHoldemEngine) -> list[AbstractActionSpec]:
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
            for raise_to in self._raise_targets(engine=engine, seat=seat):
                actions.append(("raise", raise_to))

        return actions

    def infoset_key(self, engine: NoLimitHoldemEngine, seat: int) -> str:
        return encode_engine_infoset_key(
            engine=engine,
            seat=seat,
            history_scope=self.history_scope,
            preflop_bucket_policy=self.abstraction_config.preflop_bucket_policy,
            postflop_bucket_policy=self.abstraction_config.postflop_bucket_policy,
        )

    def public_state_token(self, engine: NoLimitHoldemEngine) -> str:
        return build_public_state_key(
            engine=engine,
            history_scope=self.history_scope,
        ).to_token()

    def _raise_targets(self, engine: NoLimitHoldemEngine, seat: int) -> list[int]:
        legal = engine.get_legal_actions(seat)
        if legal.min_raise_to is None or legal.max_raise_to is None:
            return []

        min_to = legal.min_raise_to
        max_to = legal.max_raise_to
        if max_to < min_to:
            return []

        targets: set[int] = set()

        if engine.street == Street.PREFLOP:
            for multiplier in self.abstraction_config.preflop_raise_multipliers:
                raw = int(round(multiplier * engine.big_blind))
                targets.add(_clamp(raw, low=min_to, high=max_to))
        else:
            fractions = self._postflop_pot_raise_fractions(engine.street)
            base_pot = max(engine.total_pot, engine.big_blind)
            for fraction in fractions:
                raise_size = int(round(fraction * base_pot))
                raise_to = engine.current_bet + max(raise_size, engine.last_full_raise_size)
                targets.add(_clamp(raise_to, low=min_to, high=max_to))

        if self.abstraction_config.include_all_in:
            targets.add(max_to)

        ordered = sorted(
            target
            for target in targets
            if min_to <= target <= max_to and target > engine.current_bet
        )

        if self.abstraction_config.max_raise_actions <= 0:
            return ordered

        if len(ordered) <= self.abstraction_config.max_raise_actions:
            return ordered

        return _downsample_sorted_targets(ordered, self.abstraction_config.max_raise_actions)

    def _postflop_pot_raise_fractions(self, street: Street) -> tuple[float, ...]:
        if street == Street.FLOP and self.abstraction_config.flop_pot_raise_fractions is not None:
            return self.abstraction_config.flop_pot_raise_fractions
        if street == Street.TURN and self.abstraction_config.turn_pot_raise_fractions is not None:
            return self.abstraction_config.turn_pot_raise_fractions
        if street == Street.RIVER and self.abstraction_config.river_pot_raise_fractions is not None:
            return self.abstraction_config.river_pot_raise_fractions
        return self.abstraction_config.postflop_pot_raise_fractions


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

    # Repeated rounded indices can leave fewer than `limit` targets.
    if len(out) < limit:
        for value in values:
            if value in seen:
                continue
            out.append(value)
            seen.add(value)
            if len(out) == limit:
                break

    return out
