import json
from pathlib import Path
import unittest

from pluribus_ri.blueprint import BlueprintPolicy
from pluribus_ri.core import Action, NoLimitHoldemEngine
from pluribus_ri.runtime_search import (
    LeafContinuationConfig,
    NestedUnsafeSearchConfig,
    NestedUnsafeSearcher,
    SubgameStoppingRules,
    build_public_search_root,
)


_GOLDEN_PATH = Path(__file__).with_name("data") / "runtime_search_golden_v1.json"


class RuntimeSearchGoldenRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))

    def test_golden_corpus_matches_expected_search_signatures(self) -> None:
        policy = BlueprintPolicy(iteration=0, preflop_average={}, postflop_current={})
        config = _build_search_config(self.golden["config"])

        for case in self.golden["cases"]:
            with self.subTest(case=case["name"]):
                engine = _build_engine_from_case(case)
                root = build_public_search_root(engine)
                acting_seat = int(engine.to_act)  # type: ignore[arg-type]

                result = NestedUnsafeSearcher(config).search(
                    root=root,
                    blueprint_policy=policy,
                    acting_seat=acting_seat,
                )
                self.assertEqual(_result_signature(engine, result), case["expected"])

    def test_fresh_searcher_is_deterministic_for_each_golden_case(self) -> None:
        policy = BlueprintPolicy(iteration=0, preflop_average={}, postflop_current={})
        config = _build_search_config(self.golden["config"])

        for case in self.golden["cases"]:
            with self.subTest(case=case["name"]):
                engine_a = _build_engine_from_case(case)
                root_a = build_public_search_root(engine_a)
                seat_a = int(engine_a.to_act)  # type: ignore[arg-type]
                sig_a = _result_signature(
                    engine_a,
                    NestedUnsafeSearcher(config).search(
                        root=root_a,
                        blueprint_policy=policy,
                        acting_seat=seat_a,
                    ),
                )

                engine_b = _build_engine_from_case(case)
                root_b = build_public_search_root(engine_b)
                seat_b = int(engine_b.to_act)  # type: ignore[arg-type]
                sig_b = _result_signature(
                    engine_b,
                    NestedUnsafeSearcher(config).search(
                        root=root_b,
                        blueprint_policy=policy,
                        acting_seat=seat_b,
                    ),
                )
                self.assertEqual(sig_a, sig_b)


def _build_search_config(raw: dict[str, object]) -> NestedUnsafeSearchConfig:
    stop = raw["stopping_rules"]
    leaf = raw["leaf_continuation"]
    if not isinstance(stop, dict) or not isinstance(leaf, dict):
        raise ValueError("golden config is malformed")

    return NestedUnsafeSearchConfig(
        random_seed=int(raw["random_seed"]),
        freeze_own_actions=bool(raw["freeze_own_actions"]),
        insert_off_tree_actions=bool(raw["insert_off_tree_actions"]),
        history_scope=str(raw["history_scope"]),  # type: ignore[arg-type]
        stopping_rules=SubgameStoppingRules(
            min_cfr_iterations=int(stop["min_cfr_iterations"]),
            max_cfr_iterations=int(stop["max_cfr_iterations"]),
            max_nodes_touched=int(stop["max_nodes_touched"]),
            max_wallclock_ms=int(stop["max_wallclock_ms"]),
            leaf_max_depth=int(stop["leaf_max_depth"]),
        ),
        leaf_continuation=LeafContinuationConfig(
            rollout_count=int(leaf["rollout_count"]),
            max_actions_per_rollout=int(leaf["max_actions_per_rollout"]),
            random_seed=int(leaf["random_seed"]),
        ),
    )


def _build_engine_from_case(case: dict[str, object]) -> NoLimitHoldemEngine:
    engine = NoLimitHoldemEngine(seed=int(case["seed"]))
    engine.start_hand()

    script = case["script"]
    if not isinstance(script, list):
        raise ValueError("case script is malformed")
    for raw_step in script:
        if not isinstance(raw_step, list) or not raw_step:
            raise ValueError("invalid script step")
        _apply_step(engine, raw_step)
        if engine.hand_complete:
            break

    if engine.hand_complete or engine.to_act is None:
        raise AssertionError(f"golden case {case['name']} does not end in an active decision state")
    return engine


def _apply_step(engine: NoLimitHoldemEngine, step: list[object]) -> None:
    kind = str(step[0])
    legal = engine.get_legal_actions()

    if kind == "match":
        if legal.can_check:
            engine.apply_action(Action(kind="check"))
            return
        if legal.call_amount > 0:
            engine.apply_action(Action(kind="call"))
            return
        raise AssertionError("script requested match action where neither check nor call is legal")

    if kind in {"fold", "call", "check"}:
        engine.apply_action(Action(kind=kind))
        return

    if kind != "raise":
        raise ValueError(f"unsupported script action {kind}")

    if legal.min_raise_to is None or legal.max_raise_to is None:
        raise AssertionError("script requested raise where raise is illegal")
    if len(step) < 2:
        raise ValueError("raise step missing amount selector")

    selector = str(step[1])
    if selector == "min":
        amount = int(legal.min_raise_to)
    elif selector == "max":
        amount = int(legal.max_raise_to)
    elif selector == "mid":
        amount = (int(legal.min_raise_to) + int(legal.max_raise_to)) // 2
    elif selector.startswith("min+"):
        amount = int(legal.min_raise_to) + int(selector.split("+", maxsplit=1)[1])
        amount = min(int(legal.max_raise_to), max(int(legal.min_raise_to), amount))
    else:
        amount = int(selector)
    engine.apply_action(Action(kind="raise", amount=amount))


def _result_signature(engine: NoLimitHoldemEngine, result) -> dict[str, object]:
    return {
        "street": engine.street.value,
        "to_act": int(engine.to_act),  # type: ignore[arg-type]
        "chosen": {
            "kind": result.chosen_action.kind,
            "amount": int(result.chosen_action.amount),
        },
        "cfr_iterations": int(result.cfr_iterations),
        "stopping_reason": result.stopping_reason,
        "nodes_visited": int(result.nodes_visited),
        "off_tree_insertions": int(result.off_tree_insertions),
        "frozen_own_action_count": int(result.frozen_own_action_count),
        "root_strategy": [round(float(value), 6) for value in result.root_strategy],
        "action_values": [
            {
                "kind": value.action.kind,
                "amount": int(value.action.amount),
                "mean_utility": round(float(value.mean_utility), 6),
                "strategy_probability": round(float(value.strategy_probability), 6),
            }
            for value in result.action_values
        ],
    }


if __name__ == "__main__":
    unittest.main()
