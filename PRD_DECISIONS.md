# PRD Decisions Log

This file records implementation choices for gaps that are intentionally unspecified in Brown and Sandholm (2019) and listed in the project PRD.

## Current decisions (Phase 0-5)

- Scope started: PRD Phase 0 (core six-player no-limit hold'em engine, legal actions, hand evaluation, deterministic replay tests).
- Core engine module: `pluribus_ri/core/engine.py`
- Hand evaluator: `eval7` 7-card evaluator.
- Determinism: fixed RNG seed + serializable deck order in hand history.
- Replay format: full action sequence with per-action seat/kind/amount and full deck order.
- Phase 1 scaffold modules are now in place.
- Infoset/public-state keying: deterministic token schema in `pluribus_ri/abstraction/infoset.py`.
- Regret storage: lazy infoset allocation with 4-byte signed integer regrets and configurable regret floor (default `-310_000_000`).
- Solver core: external-sampling Linear MCCFR traversal implemented over a generic extensive-game interface.
- CFR weighting policy (v1 scaffold): iteration-weighted regret and average-strategy updates (`linear_weighting=True`).
- Negative-regret pruning (v1 scaffold): configurable threshold and exploration probability (`explore_all_actions_probability`).
- Engine state indexing policy: current-round lossless public-state token includes street, board, to-act seat, pot, current bet, per-seat stacks/contributions/active mask, and scoped action history.
- History scope default: `street` (current round only), with optional `all` mode for debugging/research traces.
- Private bucket proxy (v1): deterministic preflop rank/suit bucket and postflop coarse hand-type + score-modulo bucket (explicit placeholder until stronger clustering phase).
- V1 action abstraction policy (adapter):
- Preflop raise candidates from big-blind multipliers `(2.0, 3.0, 5.0, 10.0)`, clamped to legal bounds.
- Postflop raise candidates from pot fractions `(0.5, 1.0, 2.0)`, clamped to legal bounds.
- All-in included as an explicit candidate, then deduplicated and downsampled by `max_raise_actions`.
- Phase 1 orchestration policy:
- Training execution uses `train_steps` with absolute iteration counting and optional per-iteration callback hooks.
- Checkpoint format: JSON payload with configs, stats, serialized regrets, and average-strategy sums.
- Snapshot policy: preflop infosets export average strategy; postflop infosets export current strategy.
- Artifact layout: `output_dir/checkpoints/checkpoint_iter_XXXXXX.json`, `output_dir/snapshots/snapshot_iter_XXXXXX.json`, and `output_dir/summary.json`.
- Phase 2 abstraction-layer builder policy:
- Dedicated builder module `pluribus_ri/abstraction/game_builder.py` owns abstract legal-action generation and infoset/public-state tokening.
- Solver adapter delegates abstraction responsibilities to the builder (`pluribus_ri/solver/nlth_game.py`) to keep traversal logic separate from abstraction logic.
- Phase 2 playable blueprint policy:
- Runtime policy source: latest strategy snapshot (`preflop_average`, `postflop_current`) exported as a playable blueprint artifact.
- Action-selection fallback: if an infoset is missing or dimension-mismatched, use uniform legal-action probabilities to guarantee end-to-end playability.
- Deterministic action mode: argmax when sampling RNG is not supplied; stochastic sampling when RNG is supplied.
- Phase 2 orchestration policy:
- Added `Phase2RunConfig` + `run_phase2_training` to run MCCFR, export playable blueprint JSON, and execute deterministic-seed self-play validation.
- Artifact layout extended with `output_dir/blueprints/blueprint_iter_XXXXXX.json`; summary includes `self_play` metrics and `zero_sum_check`.
- Phase boundary clarification:
- Phase 2 completion criterion in this repo is a playable blueprint that can train and self-play end to end.
- Blueprint strength evaluation (cross-play leagues, mbb/hand confidence intervals, AIVAT) is explicitly tracked as later evaluation work (Phase 5 in the PRD roadmap).
- Phase 3 runtime-search decisions:
- Public-root reconstruction is implemented as deterministic replay from hand start up to (but excluding) actions on the current street.
- Runtime-search root object carries both `round_start_engine` and a marginal outside-observer belief state initialized from public cards at the round start.
- Belief representation uses independent per-seat marginals over all feasible two-card combinations (1,326 preflop baseline), with a pluggable action-likelihood interface for Bayesian reweighting.
- Nested unsafe search uses iterative external-sampling MCCFR over a round-start depth-limited subgame with configurable stopping rules (iteration, node, wallclock, and leaf depth).
- Leaf values at depth cutoffs are computed with continuation strategy mixtures (`blueprint`, `fold_biased`, `call_biased`, `raise_biased`) via sampled rollout evaluation.
- Own-action freezing records prior in-round own actions as a public-token map (`FrozenOwnActionMap`) and enforces them during subgame traversal.
- Off-tree action handling uses deterministic insertion at observed off-tree raise nodes plus pseudo-harmonic translation in reciprocal raise-size space (`PseudoHarmonicRaiseTranslator`) for mapped-action fallback and analysis.
- Runtime-search optimization guardrails now include a versioned golden corpus (`tests/data/runtime_search_golden_v1.json`) with deterministic resolver signatures for fixed public-state scenarios.
- Randomized property tests cover round-prefix reconstruction, forced/insertion-map replay consistency, and pseudo-harmonic/off-tree insertion invariants under sampled legal raise ranges.
- Runtime-search benchmark interface is standardized via `run_nested_search_benchmark(...)` and CLI `python -m pluribus_ri.bench_runtime_search`, reporting p50/p95/p99 latency, node/iteration statistics, and stop-reason distributions.
- Runtime-search optimization pass 1 uses explicit engine structural cloning (`clone_for_simulation`) instead of generic `copy.deepcopy` in traversal/rollout hot loops to increase nodes-per-millisecond under fixed wallclock budgets while preserving deterministic behavior via golden-corpus regression tests.
- Phase 4 abstraction-quality decisions:
- Introduced dedicated file-backed abstraction tables config (`pluribus_ri/abstraction/tables.py`) covering history scope, raise-size tables, and preflop bucket policy.
- Action abstraction now supports street-specific postflop raise fractions (`flop_pot_raise_fractions`, `turn_pot_raise_fractions`, `river_pot_raise_fractions`) with fallback to global postflop fractions.
- Added canonical 169 preflop bucket policy (`canonical169`) as an opt-in alternative to legacy buckets; legacy remains default for backward compatibility.
- Added postflop bucket policy switch with `texture_v1` as an opt-in alternative to legacy postflop buckets, combining hand-strength bins with draw and board-texture features.
- Added offline abstraction calibration metrics (`build_postflop_bucket_calibration_report`) and analyzer CLI (`python -m pluribus_ri.analyze_abstraction`) to compare bucket compactness and score coherence across policies.
- Training CLI supports `--abstraction-config` JSON override to load raise/bucket tables directly for runs and experiments.
- Phase boundary clarification:
- Phase 4 completion criterion in this repo is met: configurable abstraction tables, canonical preflop buckets, postflop `texture_v1` buckets, and offline calibration tooling are implemented and validated.
- Comparative blueprint/policy strength evaluation across configurations (league play, mbb/hand confidence intervals, exploitability proxies) is explicitly deferred to Phase 5 evaluation work.
- Phase 5 evaluation decisions:
- Initial benchmark harness uses one-vs-field six-max cross-play: for each ordered pair `(candidate, field)`, run the candidate in each seat once (seat-rotation), with all remaining seats using the field policy.
- Reported strength metrics are per-matchup mean utility/hand, bb/hand, mbb/hand, and normal-approximation confidence intervals, plus an aggregate mbb/hand matrix across entrants.
- Evaluation runs include zero-sum diagnostics (`max_abs(sum(terminal_utilities))`) as a correctness guardrail for simulator/regression checks.
- Added exploitability-proxy mode: each candidate is evaluated against a configurable baseline pool (`uniform`, `check_fold`, `call_biased`, `raise_biased`) with seat rotation.
- Expanded baseline pool with stronger scripted opponents (`tight_aggressive`, `loose_aggressive`, `pot_odds`) and explicit oracle-style cheater probes (`cheater_weak`, `cheater_strong`) for stress-testing only.
- `cheater_weak` is intentionally unfair via opponent private-card access; `cheater_strong` additionally uses hidden future runout from deck state for near-oracle decisions.
- Proxy exploitability score is defined as `max(0, -worst_case_mbb_per_hand_vs_baseline_pool)` to capture downside against the strongest available baseline in the proxy set.
- Added optional control-variate adjusted confidence metrics: for each matchup, evaluate a matched reference lineup with a configured baseline and report adjusted stdev/CI-width reduction diagnostics.
- Added AIVAT-style action correction: per-decision legal-action value estimates (`q_hat`) are sampled via short rollout baselines and applied as `q_hat(a_taken) - E_pi[q_hat(a)]` corrections.
- Current AIVAT scope in this repo is action-correction only; full chance/imaginary-observation correction terms are still pending.
- Core verification baseline (simple):
- Engine: randomized legal-action traversal checks legal-action executability, illegal-action rejection, and chip-conservation invariants.
- Solver-core: single-decision toy game verifies regret ranking and strong preference toward the utility-dominant action.

## Open decisions (explicitly deferred)

- Postflop bucket feature pipeline and centroids.
- Stronger exploitability proxy beyond heuristic baseline pool (for example restricted best-response approximators or richer scripted pools).
- Full AIVAT chance/imaginary-observation correction pipeline and estimator calibration beyond current action-correction implementation.
- Search budget policy and MC-CFR vs vectorized-CFR switch heuristic.
- Compression format for blueprint and continuation strategies.
- Parallel scheduler layout for training and runtime search.
