from dataclasses import dataclass, field
import random

from pluribus_ri.abstraction import (
    ActionAbstractionConfig,
    HistoryScope,
    NLTHAbstractGameBuilder,
    build_public_state_key,
)
from pluribus_ri.blueprint import BlueprintPolicy
from pluribus_ri.core import Action, NoLimitHoldemEngine
from pluribus_ri.solver import (
    ExternalSamplingLinearMCCFR,
    LazyIntRegretTable,
    MCCFRConfig,
    NLTHAbstractAction,
    NLTHAbstractGameState,
)

from .action_translation import (
    PseudoHarmonicRaiseTranslator,
    insert_off_tree_action,
)
from .continuation import ContinuationLeafEvaluator, LeafContinuationConfig
from .public_root import PublicSearchRoot
from .stopping import SubgameSearchStopController, SubgameStoppingRules


@dataclass(frozen=True)
class NestedUnsafeSearchConfig:
    random_seed: int = 0
    freeze_own_actions: bool = True
    insert_off_tree_actions: bool = True
    history_scope: HistoryScope = "street"
    stopping_rules: SubgameStoppingRules = field(default_factory=SubgameStoppingRules)
    leaf_continuation: LeafContinuationConfig = field(default_factory=LeafContinuationConfig)
    abstraction_config: ActionAbstractionConfig = field(default_factory=ActionAbstractionConfig)
    linear_weighting: bool = True


@dataclass(frozen=True)
class ActionValueEstimate:
    action: Action
    mean_utility: float
    strategy_probability: float


@dataclass(frozen=True)
class NestedUnsafeSearchResult:
    chosen_action: Action
    chosen_action_value: float
    action_values: tuple[ActionValueEstimate, ...]
    rollouts_run: int
    nodes_visited: int
    frozen_own_action_count: int
    off_tree_insertions: int
    cfr_iterations: int
    stopping_reason: str
    root_strategy: tuple[float, ...]


@dataclass(frozen=True)
class FrozenOwnActionMap:
    actions_by_public_token: dict[str, Action]

    @classmethod
    def from_public_root(
        cls,
        root: PublicSearchRoot,
        seat: int,
        history_scope: HistoryScope = "street",
    ) -> "FrozenOwnActionMap":
        current_street = root.street
        replay = root.round_start_engine.clone_for_simulation()
        mapping: dict[str, Action] = {}

        street_actions = [
            action
            for action in root.current_engine.action_log
            if action.street == current_street
        ]

        for observed in street_actions:
            if replay.to_act != observed.seat:
                raise RuntimeError("current-round replay desynced while building own-action freeze map")

            if observed.seat == seat:
                token = build_public_state_key(replay, history_scope=history_scope).to_token()
                mapping[token] = Action(kind=observed.kind, amount=int(observed.amount))

            replay.apply_action(Action(kind=observed.kind, amount=int(observed.amount)))

        return cls(actions_by_public_token=mapping)

    def lookup(self, public_state_token: str) -> Action | None:
        return self.actions_by_public_token.get(public_state_token)

    @property
    def count(self) -> int:
        return len(self.actions_by_public_token)


class NestedUnsafeSearcher:
    """
    Phase-3 nested-unsafe subgame resolver.

    Search is rebuilt from the start of the current betting round and solved with
    iterative external-sampling MCCFR over a depth-limited subgame. Leaf values
    are provided by continuation-strategy rollouts.
    """

    def __init__(self, config: NestedUnsafeSearchConfig | None = None) -> None:
        self.config = config or NestedUnsafeSearchConfig()
        self.builder = NLTHAbstractGameBuilder(
            abstraction_config=self.config.abstraction_config,
            history_scope=self.config.history_scope,
        )
        self.translator = PseudoHarmonicRaiseTranslator()
        self.rng = random.Random(self.config.random_seed)

    def search(
        self,
        root: PublicSearchRoot,
        blueprint_policy: BlueprintPolicy,
        acting_seat: int,
    ) -> NestedUnsafeSearchResult:
        if root.current_engine.to_act != acting_seat:
            raise ValueError("acting_seat must match current engine to_act")

        frozen = FrozenOwnActionMap(actions_by_public_token={})
        if self.config.freeze_own_actions:
            frozen = FrozenOwnActionMap.from_public_root(
                root=root,
                seat=acting_seat,
                history_scope=self.config.history_scope,
            )

        prefix = self._build_current_round_prefix_maps(root)

        leaf_config = self.config.leaf_continuation
        # Derive deterministic but decorrelated evaluator seed from search RNG.
        evaluator = ContinuationLeafEvaluator(
            blueprint_policy=blueprint_policy,
            config=LeafContinuationConfig(
                rollout_count=leaf_config.rollout_count,
                max_actions_per_rollout=leaf_config.max_actions_per_rollout,
                random_seed=self.rng.randint(0, 2**31 - 1),
                strategy_mix=leaf_config.strategy_mix,
                history_scope=self.config.history_scope,
                abstraction_config=self.config.abstraction_config,
            ),
        )

        root_stacks = tuple(root.current_engine._hand_starting_stacks)
        template_state = DepthLimitedSubgameState(
            engine=root.round_start_engine,
            root_stacks=root_stacks,
            acting_seat=acting_seat,
            depth=0,
            builder=self.builder,
            leaf_evaluator=evaluator,
            leaf_max_depth=self.config.stopping_rules.leaf_max_depth,
            forced_actions=prefix.forced_actions,
            inserted_actions=prefix.inserted_actions,
            frozen_actions=frozen.actions_by_public_token,
        )

        table = LazyIntRegretTable()
        trainer = ExternalSamplingLinearMCCFR(
            regret_table=table,
            config=MCCFRConfig(
                iterations=1,
                random_seed=self.config.random_seed,
                linear_weighting=self.config.linear_weighting,
            ),
        )
        stopper = SubgameSearchStopController(self.config.stopping_rules)
        iterations = 0
        stopping_reason = "continue"
        while True:
            iterations += 1
            trainer.train_steps(
                root_state_factory=template_state.clone,
                num_players=root.current_engine.num_players,
                iterations=1,
            )
            should_stop, reason = stopper.should_stop(
                iterations=iterations,
                nodes_touched=trainer.stats.nodes_touched,
            )
            stopping_reason = reason
            if should_stop:
                break

        current_state = DepthLimitedSubgameState(
            engine=root.current_engine,
            root_stacks=root_stacks,
            acting_seat=acting_seat,
            depth=0,
            builder=self.builder,
            leaf_evaluator=evaluator,
            leaf_max_depth=self.config.stopping_rules.leaf_max_depth,
            forced_actions={},
            inserted_actions=prefix.inserted_actions,
            frozen_actions=frozen.actions_by_public_token,
        )
        legal_actions = list(current_state.legal_actions())
        if not legal_actions:
            raise RuntimeError("no legal actions at current search state")

        infoset_key = current_state.infoset_key(acting_seat)
        strategy = table.current_strategy(infoset_key, len(legal_actions))

        estimates: list[ActionValueEstimate] = []
        for idx, abstract_action in enumerate(legal_actions):
            engine_action = abstract_action.to_engine_action()
            child_engine = root.current_engine.clone_for_simulation()
            child_engine.apply_action(engine_action)
            value = _evaluate_terminal_or_leaf(
                engine=child_engine,
                acting_seat=acting_seat,
                root_stacks=root_stacks,
                leaf_evaluator=evaluator,
            )
            estimates.append(
                ActionValueEstimate(
                    action=engine_action,
                    mean_utility=value,
                    strategy_probability=strategy[idx],
                )
            )

        chosen = max(
            estimates,
            key=lambda item: (item.strategy_probability, item.mean_utility),
        )
        return NestedUnsafeSearchResult(
            chosen_action=chosen.action,
            chosen_action_value=chosen.mean_utility,
            action_values=tuple(estimates),
            rollouts_run=evaluator.rollouts_run,
            nodes_visited=trainer.stats.nodes_touched,
            frozen_own_action_count=frozen.count,
            off_tree_insertions=prefix.off_tree_insertions,
            cfr_iterations=iterations,
            stopping_reason=stopping_reason,
            root_strategy=tuple(strategy),
        )

    def _build_current_round_prefix_maps(
        self,
        root: PublicSearchRoot,
    ) -> "_RoundPrefixMaps":
        replay = root.round_start_engine.clone_for_simulation()
        forced: dict[str, Action] = {}
        inserted: dict[str, tuple[Action, ...]] = {}
        off_tree_insertions = 0

        street_actions = [
            action
            for action in root.current_engine.action_log
            if action.street == root.street
        ]
        for observed in street_actions:
            token = self.builder.public_state_token(replay)
            observed_action = Action(kind=observed.kind, amount=int(observed.amount))

            base_actions = [a.to_engine_action() for a in _legal_abstract_actions(replay, self.builder)]
            translation = self.translator.translate(
                engine=replay,
                observed_action=observed_action,
                abstract_legal_actions=base_actions,
            )

            forced_action = translation.translated_action
            if translation.was_off_tree and self.config.insert_off_tree_actions:
                insertion = insert_off_tree_action(
                    engine=replay,
                    observed_action=observed_action,
                    abstract_legal_actions=base_actions,
                )
                if insertion.was_inserted:
                    inserted[token] = tuple(
                        action
                        for action in insertion.actions
                        if action.kind == "raise" and action.amount == observed_action.amount
                    )
                    off_tree_insertions += 1
                    forced_action = observed_action

            forced[token] = forced_action
            replay.apply_action(observed_action)

        replay_token = self.builder.public_state_token(replay)
        current_token = self.builder.public_state_token(root.current_engine)
        if replay_token != current_token:
            raise RuntimeError("round-prefix reconstruction desynced from current engine")

        return _RoundPrefixMaps(
            forced_actions=forced,
            inserted_actions=inserted,
            off_tree_insertions=off_tree_insertions,
        )


@dataclass(frozen=True)
class _RoundPrefixMaps:
    forced_actions: dict[str, Action]
    inserted_actions: dict[str, tuple[Action, ...]]
    off_tree_insertions: int


class DepthLimitedSubgameState:
    def __init__(
        self,
        engine: NoLimitHoldemEngine,
        root_stacks: tuple[int, ...],
        acting_seat: int,
        depth: int,
        builder: NLTHAbstractGameBuilder,
        leaf_evaluator: ContinuationLeafEvaluator,
        leaf_max_depth: int,
        forced_actions: dict[str, Action],
        inserted_actions: dict[str, tuple[Action, ...]],
        frozen_actions: dict[str, Action],
    ) -> None:
        self.engine = engine
        self.root_stacks = root_stacks
        self.acting_seat = acting_seat
        self.depth = depth
        self.builder = builder
        self.leaf_evaluator = leaf_evaluator
        self.leaf_max_depth = leaf_max_depth
        self.forced_actions = forced_actions
        self.inserted_actions = inserted_actions
        self.frozen_actions = frozen_actions

    def clone(self) -> "DepthLimitedSubgameState":
        return DepthLimitedSubgameState(
            engine=self.engine.clone_for_simulation(),
            root_stacks=self.root_stacks,
            acting_seat=self.acting_seat,
            depth=self.depth,
            builder=self.builder,
            leaf_evaluator=self.leaf_evaluator,
            leaf_max_depth=self.leaf_max_depth,
            forced_actions=self.forced_actions,
            inserted_actions=self.inserted_actions,
            frozen_actions=self.frozen_actions,
        )

    def is_terminal(self) -> bool:
        return self.engine.hand_complete or self.depth >= self.leaf_max_depth

    def utility(self, player: int) -> float:
        if self.engine.hand_complete:
            return float(self.engine.stacks[player] - self.root_stacks[player])
        return self.leaf_evaluator.evaluate(
            engine=self.engine,
            player=player,
            root_stacks=self.root_stacks,
        )

    def is_chance_node(self) -> bool:
        return False

    def chance_outcomes(self) -> list[tuple[NLTHAbstractAction, float]]:
        return []

    def current_player(self) -> int:
        if self.engine.to_act is None:
            raise RuntimeError("non-terminal state has no player to act")
        return int(self.engine.to_act)

    def legal_actions(self) -> list[NLTHAbstractAction]:
        if self.engine.hand_complete or self.engine.to_act is None:
            return []

        token = self.builder.public_state_token(self.engine)
        actions = [action.to_engine_action() for action in _legal_abstract_actions(self.engine, self.builder)]

        inserted = self.inserted_actions.get(token, ())
        for action in inserted:
            if _is_legal_action(self.engine, action) and not _contains_action(actions, action):
                actions.append(action)
        actions = _normalized_action_order(actions)

        forced = self.forced_actions.get(token)
        if forced is not None:
            if not _contains_action(actions, forced):
                if _is_legal_action(self.engine, forced):
                    actions.append(forced)
                    actions = _normalized_action_order(actions)
                else:
                    raise RuntimeError("forced prefix action is illegal in subgame state")
            return [_as_abstract_action(forced)]

        if int(self.engine.to_act) == self.acting_seat:
            frozen = self.frozen_actions.get(token)
            if frozen is not None and _contains_action(actions, frozen):
                return [_as_abstract_action(frozen)]

        return [_as_abstract_action(action) for action in actions]

    def child(self, action: NLTHAbstractAction) -> "DepthLimitedSubgameState":
        next_engine = self.engine.clone_for_simulation()
        next_engine.apply_action(action.to_engine_action())
        return DepthLimitedSubgameState(
            engine=next_engine,
            root_stacks=self.root_stacks,
            acting_seat=self.acting_seat,
            depth=self.depth + 1,
            builder=self.builder,
            leaf_evaluator=self.leaf_evaluator,
            leaf_max_depth=self.leaf_max_depth,
            forced_actions=self.forced_actions,
            inserted_actions=self.inserted_actions,
            frozen_actions=self.frozen_actions,
        )

    def infoset_key(self, player: int) -> str:
        return self.builder.infoset_key(self.engine, player)


def _evaluate_terminal_or_leaf(
    engine: NoLimitHoldemEngine,
    acting_seat: int,
    root_stacks: tuple[int, ...],
    leaf_evaluator: ContinuationLeafEvaluator,
) -> float:
    if engine.hand_complete:
        return float(engine.stacks[acting_seat] - root_stacks[acting_seat])
    return leaf_evaluator.evaluate(
        engine=engine,
        player=acting_seat,
        root_stacks=root_stacks,
    )


def _legal_abstract_actions(
    engine: NoLimitHoldemEngine,
    builder: NLTHAbstractGameBuilder,
) -> list[NLTHAbstractAction]:
    state = NLTHAbstractGameState(
        engine=engine,
        root_stacks=tuple(engine._hand_starting_stacks),
        abstraction_builder=builder,
    )
    return list(state.legal_actions())


def _is_legal_action(engine: NoLimitHoldemEngine, action: Action) -> bool:
    legal = engine.get_legal_actions()
    if action.kind == "fold":
        return legal.can_fold
    if action.kind == "check":
        return legal.can_check
    if action.kind == "call":
        return legal.call_amount > 0
    if action.kind == "raise":
        if legal.min_raise_to is None or legal.max_raise_to is None:
            return False
        return legal.min_raise_to <= action.amount <= legal.max_raise_to
    return False


def _contains_action(actions: list[Action], action: Action) -> bool:
    return any(existing.kind == action.kind and existing.amount == action.amount for existing in actions)


def _as_abstract_action(action: Action) -> NLTHAbstractAction:
    return NLTHAbstractAction(kind=action.kind, amount=action.amount)


def _normalized_action_order(actions: list[Action]) -> list[Action]:
    non_raises = [action for action in actions if action.kind != "raise"]
    raises = sorted(
        [action for action in actions if action.kind == "raise"],
        key=lambda action: action.amount,
    )
    return non_raises + raises
