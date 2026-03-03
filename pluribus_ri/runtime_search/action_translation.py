from dataclasses import dataclass
from typing import Protocol, Sequence

from pluribus_ri.core import Action, NoLimitHoldemEngine


class OffTreeActionTranslator(Protocol):
    """Interface for mapping observed off-tree actions into abstract actions."""

    def translate(
        self,
        engine: NoLimitHoldemEngine,
        observed_action: Action,
        abstract_legal_actions: Sequence[Action],
    ) -> "ActionTranslationResult":
        ...


@dataclass(frozen=True)
class ActionTranslationResult:
    translated_action: Action
    was_off_tree: bool
    distance: int
    mixed_strategy: tuple[tuple[Action, float], ...] = ()


@dataclass(frozen=True)
class OffTreeInsertionResult:
    actions: tuple[Action, ...]
    was_inserted: bool


class NearestRaiseActionTranslator:
    """
    Simple off-tree hook for v1 scaffolding.

    For raise actions not present in the abstract set, this maps to the nearest
    abstract raise target. This intentionally serves as a placeholder until a
    pseudo-harmonic mapping implementation is added.
    """

    def translate(
        self,
        engine: NoLimitHoldemEngine,
        observed_action: Action,
        abstract_legal_actions: Sequence[Action],
    ) -> ActionTranslationResult:
        del engine

        if observed_action.kind != "raise":
            return ActionTranslationResult(
                translated_action=observed_action,
                was_off_tree=False,
                distance=0,
                mixed_strategy=((observed_action, 1.0),),
            )

        raise_actions = [action for action in abstract_legal_actions if action.kind == "raise"]
        if not raise_actions:
            return ActionTranslationResult(
                translated_action=observed_action,
                was_off_tree=False,
                distance=0,
                mixed_strategy=((observed_action, 1.0),),
            )

        exact = next((action for action in raise_actions if action.amount == observed_action.amount), None)
        if exact is not None:
            return ActionTranslationResult(
                translated_action=exact,
                was_off_tree=False,
                distance=0,
                mixed_strategy=((exact, 1.0),),
            )

        nearest = min(raise_actions, key=lambda action: abs(action.amount - observed_action.amount))
        return ActionTranslationResult(
            translated_action=nearest,
            was_off_tree=True,
            distance=abs(nearest.amount - observed_action.amount),
            mixed_strategy=((nearest, 1.0),),
        )


class PseudoHarmonicRaiseTranslator:
    """
    Pseudo-harmonic off-tree raise mapper.

    The translation is computed in reciprocal raise-size space and produces a
    mixed action over the bracketing abstract raise sizes. This keeps mapping
    behavior stable at large bet sizes while preserving locality.
    """

    def translate(
        self,
        engine: NoLimitHoldemEngine,
        observed_action: Action,
        abstract_legal_actions: Sequence[Action],
    ) -> ActionTranslationResult:
        del engine

        if observed_action.kind != "raise":
            return ActionTranslationResult(
                translated_action=observed_action,
                was_off_tree=False,
                distance=0,
                mixed_strategy=((observed_action, 1.0),),
            )

        raise_actions = sorted(
            [action for action in abstract_legal_actions if action.kind == "raise"],
            key=lambda action: action.amount,
        )
        if not raise_actions:
            return ActionTranslationResult(
                translated_action=observed_action,
                was_off_tree=False,
                distance=0,
                mixed_strategy=((observed_action, 1.0),),
            )

        exact = next((action for action in raise_actions if action.amount == observed_action.amount), None)
        if exact is not None:
            return ActionTranslationResult(
                translated_action=exact,
                was_off_tree=False,
                distance=0,
                mixed_strategy=((exact, 1.0),),
            )

        if observed_action.amount <= raise_actions[0].amount:
            edge = raise_actions[0]
            return ActionTranslationResult(
                translated_action=edge,
                was_off_tree=True,
                distance=abs(edge.amount - observed_action.amount),
                mixed_strategy=((edge, 1.0),),
            )

        if observed_action.amount >= raise_actions[-1].amount:
            edge = raise_actions[-1]
            return ActionTranslationResult(
                translated_action=edge,
                was_off_tree=True,
                distance=abs(edge.amount - observed_action.amount),
                mixed_strategy=((edge, 1.0),),
            )

        lower = raise_actions[0]
        upper = raise_actions[-1]
        for idx in range(len(raise_actions) - 1):
            lo = raise_actions[idx]
            hi = raise_actions[idx + 1]
            if lo.amount <= observed_action.amount <= hi.amount:
                lower = lo
                upper = hi
                break

        weight_lower, weight_upper = _pseudo_harmonic_weights(
            observed=float(observed_action.amount),
            lower=float(lower.amount),
            upper=float(upper.amount),
        )
        translated = lower if weight_lower >= weight_upper else upper
        return ActionTranslationResult(
            translated_action=translated,
            was_off_tree=True,
            distance=abs(translated.amount - observed_action.amount),
            mixed_strategy=((lower, weight_lower), (upper, weight_upper)),
        )


def insert_off_tree_action(
    engine: NoLimitHoldemEngine,
    observed_action: Action,
    abstract_legal_actions: Sequence[Action],
) -> OffTreeInsertionResult:
    """
    Insert a legal off-tree raise into the action list at the current node.

    Non-raise actions or raises already present are returned unchanged.
    """

    base = list(abstract_legal_actions)
    if observed_action.kind != "raise":
        return OffTreeInsertionResult(actions=tuple(base), was_inserted=False)
    if not _is_legal_raise(engine, observed_action):
        return OffTreeInsertionResult(actions=tuple(base), was_inserted=False)
    if any(action.kind == "raise" and action.amount == observed_action.amount for action in base):
        return OffTreeInsertionResult(actions=tuple(base), was_inserted=False)

    non_raises = [action for action in base if action.kind != "raise"]
    raises = sorted(
        [action for action in base if action.kind == "raise"] + [observed_action],
        key=lambda action: action.amount,
    )
    return OffTreeInsertionResult(actions=tuple(non_raises + raises), was_inserted=True)


def _is_legal_raise(engine: NoLimitHoldemEngine, action: Action) -> bool:
    if action.kind != "raise":
        return False
    legal = engine.get_legal_actions()
    if legal.min_raise_to is None or legal.max_raise_to is None:
        return False
    return legal.min_raise_to <= action.amount <= legal.max_raise_to


def _pseudo_harmonic_weights(observed: float, lower: float, upper: float) -> tuple[float, float]:
    inv_observed = 1.0 / max(observed, 1.0)
    inv_lower = 1.0 / max(lower, 1.0)
    inv_upper = 1.0 / max(upper, 1.0)

    gap_lower = abs(inv_observed - inv_lower)
    gap_upper = abs(inv_observed - inv_upper)
    if gap_lower <= 0.0 and gap_upper <= 0.0:
        return (0.5, 0.5)
    if gap_lower <= 0.0:
        return (1.0, 0.0)
    if gap_upper <= 0.0:
        return (0.0, 1.0)

    raw_lower = 1.0 / gap_lower
    raw_upper = 1.0 / gap_upper
    total = raw_lower + raw_upper
    if total <= 0.0:
        return (0.5, 0.5)
    return (raw_lower / total, raw_upper / total)
