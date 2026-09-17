# Thesis methods and configuration inventory

This is the canonical inventory for explaining what was tried, why it was
tried, what happened, and whether the evidence is strong enough for the main
thesis narrative. “Implemented” never means “empirically supported.”

## MCTS configurations and methods

This is the advisor-requested methods inventory. It is deliberately organized
around the MCTS design axes rather than around the entire training pipeline.

| Axis | Configuration | Intuition | Evidence and result | Thesis readiness |
|---|---|---|---|---|
| Leaf evaluation | Historical policy rollout | Estimate a leaf by continuing with the learned policy. | Present in early git history; replaced by value-based leaf evaluation. No controlled matched coverage/runtime comparison survives. | Describe as tried architecture only unless rerun. |
| Leaf evaluation | Learned value / ENHSP blend (`use_estimator=.5`) | Combine learned generalization with an external planning estimate. | This is the current mainline MCTS estimator used by the fixed and PW campaigns. Its end-to-end coverage is well measured, but its two components have not been factorially isolated. | Main method; do not claim `.5` is optimal. |
| Leaf evaluation | Learned value only / ENHSP only | Isolate what the value head itself contributes at the leaf. | Code support exists; no canonical matched result table. | High-priority small ablation if the thesis needs a component claim. |
| Expansion | Fixed top-20 / 70 simulations | Retain broad policy support and allow search evidence to develop. | Strong FO Counters and Drone/VH-on gains, smaller Rover gains, variable/negative Counters behavior. | Thesis-ready primary method. |
| Expansion | Narrow top-5 / 20 simulations | Keep Block Grouping and Counters computationally feasible. | Block Grouping approaches policy only at long budgets; Counters can remain below policy. | Thesis-ready domain-specific compromise. |
| Expansion | PW70 (`Kmin=3`, `c=.6`, `alpha=.5`, `Kmax=20`) | Start narrow and add actions only as root evidence accumulates. | Strong FO Counters and MPrime evidence, approximate Rover parity, weaker Drone/BG/Counters evidence. | Thesis-ready domain-dependent result. |
| Expansion | PW20 | Match the historical narrow 5/20 budget. | Isolated Counters recoveries; not consistently better. | Screening result; keep separate from PW70. |
| Expansion | Drone PW `Kmin=2` / `Kmin=3` | Test the coverage–tree-size trade-off. | `Kmin=2` cut trees/runtime substantially but lost coverage; `Kmin=3` recovered to 9.5/20 versus policy 7.0 and fixed 10.5. | Thesis-ready sensitivity result. |
| Tree selection | PUCT, exploration constant `.1` | Balance prior-guided exploration and accumulated Q. | Used consistently in the main campaigns; no matched sensitivity grid proves `.1` optimal. | Main declared setting, not an optimization claim. |
| Root action selection | Maximum visits, action-index tie | Historical deterministic selection rule. | Selected Counters failures exposed arbitrary index dependence. | Baseline behavior. |
| Root action selection | Q tie-break | Let learned/search value resolve equal visits. | 0/3 on the selected Counters causal pilot. | Negative selected-case result. |
| Root action selection | Policy-prior tie-break | Preserve the learned preference when visit/Q evidence is unresolved. | Recovered 3/3 selected Counters policy successes; strict confirmation remains required. | Promising diagnostic, not yet a primary replacement. |
| Safety | SAFE-1 terminal protection | Refuse a known terminal non-goal commitment. | Repaired 2/4 selected Drone failures. | Useful incomplete repair. |
| State identity | SAFE-CONTEXT | Distinguish identical physical states with different action histories. | Explained a real aliasing mechanism but reduced coverage and increased memory. | Thesis-ready negative result. |
| Search horizon | Remaining-horizon enforcement | Prevent search below the remaining executable budget. | Nonbinding in audited runs; no observed coverage change. | Negative/non-result. |
| Execution budget | 30m / 2h / 6h recensoring | Show when expensive search gains appear. | Exact post-hoc cutoffs are available from complete six-hour per-instance logs. | Thesis-ready reporting dimension. |

The normal fixed-search simulation rule is
`clip(10 + 3 * retained_children, 10, 200)`: 20 retained children therefore
imply 70 simulations. The narrow 5/20 setup is an explicit historical override
for expensive domains; the formula would otherwise yield 25.

### Evidence still worth generating

The most defensible missing MCTS-method evidence is:

1. **Rollout versus current leaf evaluation.** Freeze checkpoint, instances,
   top-K, simulations, PUCT and wall budget. A minimum screen is rollout versus
   the current `.5` blend on Drone and FO Counters, two seeds each: eight
   evaluation tasks. A full four-arm screen (rollout, learned-only, ENHSP-only,
   `.5` blend) is sixteen tasks. Report coverage, leaf-evaluation time, total
   runtime and generated nodes.
2. **Estimator-component ablation.** If the thesis claims that the learned
   value is useful inside MCTS, compare learned-only, ENHSP-only and the `.5`
   blend after the separate value-head quality audit. Do not run this merely to
   defend a default.
3. **PUCT sensitivity.** Only needed if claiming `.1` is preferable, rather
   than declaring it as the frozen setting. A small validation-only grid should
   precede any test evaluation.

These are proposals, not submitted work. The fixed/PW, SAFE-1, context,
horizon and selected tie-break results already have enough evidence to discuss
what succeeded, failed, and why.

## Auxiliary training and selection methodology

| Method | Intuition | Evidence | Thesis status |
|---|---|---|---|
| Stage-1 ENHSP imitation | Retain the supervised planner-imitation procedure used to train the original Numeric ASNets. | The validation-selected Stage-1 policy is the baseline for every primary RQ. | Original-paper baseline, not a newly introduced thesis method. |
| Validation-selected checkpoints | Select a deployable model without test-set knowledge. | Primary reporting is validation-led only. MPrime required Phase B/C because the original validator saturated. | Main method; disclose MPrime correction. |
| Stage-2 MCTS-target refinement | Train on search visit distributions to improve beyond imitation. | No general Holm-significant policy-only gain; outcomes are domain-specific and one TPP seed collapsed. | Main mixed/negative result. |
| Constant Stage-1 KL anchor | Discourage destructive drift during Stage 2. | TPP revealed that the historical current-policy KL forward had dropout active. Under one frozen replay/RNG stream, legacy semantics scored 11/20 and deterministic-current semantics 20/20; stable control stayed 20/20. | Main implementation finding; multi-RNG confirmation live. |
| Value-head on/off factorial | Test whether learned state value helps training or search. | No reliable general Stage-2 training benefit; strong MCTS interaction in Drone, weak/neutral elsewhere. | Main RQ3/RQ4 design. |

## Supporting methodological assurance

The CPU-determinism audit observed checksum differences across CPU families,
but the audited actions and outcomes were unchanged.  This supports treating
the reported search outcomes as method evidence rather than hardware-family
artifacts.  All MCTS refinements and their results are listed once in the main
MCTS inventory above.

## Auxiliary training safeguards

| Configuration | Intuition | Result | Status |
|---|---|---|---|
| Adaptive target-KL coefficient | Increase anchoring when observed KL becomes too high. | Never activated because the target statistic and monitored statistic were on mismatched scales. | Completed negative activation screen. |
| Adam rollback / LR backtracking | Reject an excessive update, restore network and optimizer state, lower LR and retry. | Apparent selected-seed rescue was confounded by deterministic-current KL; exact-RNG inactive arms also stayed 20/20. | Held; no isolated guard benefit. |
| Deterministic-current anchor KL | Keep replay training stochastic but compare deterministic current and Stage-1 anchor policies. | Selected frozen TPP stream: 20/20 versus 11/20 for dropout-current semantics. | Mechanism result; multi-RNG confirmation live. |
| MPrime validation redesign | Replace a saturated validator with harder frozen alternatives. | Phase B removed saturation; Phase C did not improve selector quality enough; Phase-B-A is final. | Methodological contribution. |

## Historical prototypes and evidence gaps

The early MCTS implementation used policy rollouts from leaves. Commit
`50835e09` introduced value-based leaf evaluation. This is distinct from the
Stage-1 teacher’s “ROLLOUT” experience mode. The git message says value-based
MCTS seemed slow, but there is no controlled rollout-versus-value comparison.
It may be described as an architectural transition, not as evidence that one
was better. A thesis-grade comparison would hold checkpoint, width,
simulations, PUCT and wall budget fixed and report coverage plus leaf-evaluation
runtime.

Other real code options lacking current matched evidence are: one-hot MCTS
targets (`8f69cc4b`), skipping failed search episodes, HER, goal-path
reconstruction, extra-tree-state/visit-power sampling, repeated optimization
from one exploration, heuristic bootstrapping, estimator decay (`779357ae`),
raw-value minimization (`84b2d60a`), policy sampling, epsilon/temperature and
duplicate penalties, corruption tests, removed PUCT variants (`0c5f6734`), and
parallel node generation. These belong in historical design or future work,
not the results chapter.

## Held designs

- `(state, remaining horizon)` SAFE2: no demonstrated contamination and high
  memory risk.
- Path-batched multi-node PW: changes one-path/one-backup semantics.
- Action-count input ablation: requires fresh training.
- Original-paper stopping replication: compatibility experiment, not run.
- PUCT / estimator factorial: needed before claiming current `c=.1` and blend
  `.5` are optimal.
- Fully guarded Stage 2: gate behind the KL-semantics susceptibility result.
- Fresh hard-30-minute jobs: unnecessary when complete six-hour logs permit
  exact deterministic post-hoc recensoring; needed only for censored timing or
  whole-job resource claims.

## Internal git provenance

`50835e09` rollout/value transition; `25ba839b` value head; `7dc12ed9` mixed
replay and gradient clipping; `0c2570e9` one extraction per episode;
`d987a1c0` requeue-safe KL anchor; `0a51ddc0` final MCTS evaluation;
`fac11cb0` PW/horizon; `c6bed543` SAFE-1; `9526dc0f` context-aware nodes;
`320a2eec` root tie-break; `96a6d02d` adaptive KL; `38a21b33` validation
stopping; `84b2d60a` raw-value minimization; `8f69cc4b` one-hot targets;
`779357ae` estimator decay; `0c5f6734` removed selector variants.
