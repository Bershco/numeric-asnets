# Thesis methods and configuration inventory

This is the canonical inventory for explaining what was tried, why it was
tried, what happened, and whether the evidence is strong enough for the main
thesis narrative. “Implemented” never means “empirically supported.”

## Main methods

| Method | Intuition | Evidence | Thesis status |
|---|---|---|---|
| Stage-1 ENHSP imitation | Start from planner demonstrations rather than learn planning from scratch. | The validation-selected Stage-1 policy is the baseline for every primary RQ. | Main method. |
| Validation-selected checkpoints | Select a deployable model without test-set knowledge. | Primary reporting is validation-led only. MPrime required Phase B/C because the original validator saturated. | Main method; disclose MPrime correction. |
| Stage-2 MCTS-target refinement | Train on search visit distributions to improve beyond imitation. | No general Holm-significant policy-only gain; outcomes are domain-specific and one TPP seed collapsed. | Main mixed/negative result. |
| Constant Stage-1 KL anchor | Discourage destructive drift during Stage 2. | TPP revealed that the historical current-policy KL forward had dropout active. Under one frozen replay/RNG stream, legacy semantics scored 11/20 and deterministic-current semantics 20/20; stable control stayed 20/20. | Main implementation finding; multi-RNG confirmation live. |
| Fixed MCTS, normal 20/70 | Search beyond greedy policy with broad policy-guided branching. | Strong FO Counters and Drone/VH-on gains, smaller Rover gains, near-neutral MPrime fixed search, negative/variable Counters behavior. | Main inference method. |
| Fixed MCTS, narrow 5/20 | Make expensive Block Grouping/Counters search computationally possible. | Block Grouping needs long budgets to approach policy; Counters can underperform policy. | Main domain-specific compromise, not universal default. |
| Value-head on/off factorial | Test whether learned state value helps training or search. | No reliable general Stage-2 training benefit; strong MCTS interaction in Drone, weak/neutral elsewhere. | Main RQ3/RQ4 design. |

The normal fixed-search simulation rule is
`clip(10 + 3 * retained_children, 10, 200)`: 20 retained children therefore
imply 70 simulations. The narrow 5/20 setup is an explicit historical override
for expensive domains; the formula would otherwise yield 25.

## Search refinements and diagnostics

| Configuration | Intuition | Result | Status |
|---|---|---|---|
| PW70 (`Kmin=3`, `c=.6`, `alpha=.5`, `Kmax=20`) | Expand few actions initially and widen only when visits justify it. | Strong FO Counters and MPrime results; Rover approximate parity; weaker Drone, Block Grouping and Counters results. | Successful but domain-dependent. |
| PW20 | Match the narrow 5/20 compute budget. | Isolated Counters recoveries, no consistent universal benefit. | Screening evidence; never pool with PW70. |
| Drone PW `Kmin=2` | Aggressively reduce branching and tree size. | Very large node/runtime savings but significant coverage loss versus fixed top-20. | Negative coverage / positive efficiency result. |
| Drone PW `Kmin=3` | Give PW a broader initial policy-supported base. | 9.5/20 versus policy 7.0 and fixed 10.5; better runtime tail but still lost fixed successes. | Partial repair. |
| SAFE-1 terminal protection | Avoid committing to a known terminal non-goal child. | Repaired 2/4 selected Drone failures. | Useful, incomplete diagnostic. |
| Policy-prior root tie-break | Use learned prior instead of action-array order when visits tie and Q is uninformative. | Selected Counters VH-off: action ID 0/3, Q 0/3, policy prior 3/3. | Promising causal pilot; strict confirmation still gates any primary change. |
| Q tie-break | Resolve equal visits using Q. | 0/3 in the same selected Counters pilot. | Rejected selected-case alternative. |
| SAFE-CONTEXT | Separate identical physical states with different action-history inputs. | Reduced coverage and increased memory; rejected. | Negative result. |
| Remaining-horizon enforcement | Prevent search deeper than executable remaining horizon. | Nonbinding in completed audits; no coverage change. | Negative/non-result. |
| CPU determinism audit | Test whether hardware numerical differences explain historical discrepancies. | Checksums changed across CPU families, but audited actions and outcomes did not. | Methodological assurance. |

## Training safeguards

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
