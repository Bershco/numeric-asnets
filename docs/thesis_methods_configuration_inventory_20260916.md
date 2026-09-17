# Thesis methods and configuration inventory

This is the canonical inventory for explaining what was tried, why it was
tried, what happened, and whether the evidence is strong enough for the main
thesis narrative. “Implemented” never means “empirically supported.”

## MCTS configurations and methods

This is the advisor-requested methods inventory. It separates genuine method
families from parameter studies, auxiliary safeguards and reporting choices.
Calling all of these “configurations” would exaggerate the number of distinct
methods and obscure what was actually learned.

### Primary method families

| Method family | Variant | Intuition | Evidence and result | Evidence class |
|---|---|---|---|---|
| Leaf evaluation | Historical policy rollout | Estimate a leaf by continuing with the learned policy. | Present in early git history; replaced by value/heuristic leaf evaluation. No surviving controlled comparison. | C: implemented historical method; no outcome claim yet. |
| Leaf evaluation | Network value (when VH is enabled) / ENHSP estimator blend (`use_estimator=.5`) | Combine learned value generalization, when available, with an external planning estimate. VH-off cells do not emit a learned value. | Current mainline estimator setting in fixed and PW campaigns. End-to-end results exist, but the estimator components have not been factorially isolated. | A for the end-to-end method; C for claims about why the blend helps. |
| Leaf evaluation | Learned-only / ENHSP-only | Isolate the learned and heuristic components. | Code support exists; no canonical matched result table. | C: implemented options; rerun only if making a component claim. |
| Expansion | Fixed expansion | Retain a fixed policy-ranked action set throughout search. | Strong FO Counters and Drone/VH-on gains, smaller Rover gains, variable/negative Counters behavior. | A: replicated primary method. |
| Expansion | Progressive widening | Start narrow and admit actions as evidence accumulates. | Strong FO Counters evidence, promising live MPrime lower bounds, approximate Rover parity, and weaker Drone/Block Grouping/Counters evidence. | A where ten-seed confirmation is complete; B/live where it is not. |

The normal and narrow budgets are **parameterizations of fixed search**, not
separate method families. Likewise PW20 and PW70 are simulation-budget
variants of progressive widening, not separate methods:

| Parameter study | Result | Correct thesis use |
|---|---|---|
| Fixed top-20 / 70 simulations versus top-5 / 20 simulations | The broad setting is stronger where affordable; 5/20 makes expensive Block Grouping/Counters evaluations feasible but can lose coverage. | Computational trade-off, not a new algorithm. |
| PW20 versus PW70 | PW20 produced isolated Counters recoveries; PW70 is the standard matched normal-search budget, strongly positive in FO Counters and promising but still live in MPrime. | Budget sensitivity; never pool the two. |
| Drone PW `Kmin=2` versus `Kmin=3` | `Kmin=2` greatly reduced tree/runtime but lost coverage; `Kmin=3` reached 9.5/20 versus policy 7.0 and fixed 10.5. | Hyperparameter-optimization pilot, not a method-family comparison. |
| PUCT exploration constant `.1` | Frozen main setting; no controlled evidence that it is optimal. | Declare it, do not call it optimal. A test-set grid would be descriptive post-hoc sensitivity, not valid tuning. |

### Auxiliary mechanisms and safeguards

| Mechanism | Intuition | Evidence and result | Evidence class / thesis use |
|---|---|---|---|
| Maximum visits, action-index tie | Historical deterministic root selection. | Selected Counters failures exposed arbitrary index dependence. | Baseline behavior. |
| Q tie-break | Let learned/search value resolve equal visits. | 0/3 on the selected Counters causal pilot. | B: selected-case negative pilot. |
| Policy-prior tie-break | Preserve the learned preference when visit/Q evidence is unresolved. | Recovered 3/3 selected Counters policy successes; strict confirmation is unfinished. | B: promising selected-case mechanism, not a primary replacement. |
| SAFE-1 terminal protection | Refuse a known terminal non-goal commitment. | Repaired 2/4 selected Drone failures. | B: useful selected-case pilot; broaden before an effectiveness claim. |
| SAFE-CONTEXT | Distinguish identical physical states with different action histories. | Exposed real aliasing, but the selected screen reduced coverage and increased memory. | B: negative development evidence, not a replicated main result. |
| Remaining-horizon enforcement | Prevent search below the remaining executable budget. | Nonbinding in audited cases; no observed coverage change. | B/D: documented negative/non-result. |

The 30-minute, two-hour and six-hour cutoffs are **reporting dimensions**, not
configurations. They are exact post-hoc recensorings of the same complete
six-hour evaluations.

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
3. **PUCT sensitivity.** A validation grid may select a deployable constant,
   but cannot identify the test-distribution optimum when validation is known
   to shift structurally. If the question is simply “what changes with `c`?”,
   use a predeclared post-hoc test sensitivity analysis and call it descriptive
   robustness, not tuning. If selecting a future default, use multiple frozen,
   structurally stratified validation replicates, require a stable choice,
   freeze it, and evaluate test once.

These are proposals, not submitted work. Replicated fixed/PW results support
outcome claims. SAFE-1, context, horizon and tie-break results can already be
discussed as selected diagnostics or development decisions, but not as broad
effectiveness claims.

### Evidence standard for the thesis narrative

- **A — replicated result:** supports an outcome claim in the main results.
- **B — predeclared exploratory/selected-case pilot:** supports a mechanism or
  explains why work was expanded/stopped, but not a population-wide claim.
- **C — implemented prototype:** belongs in design history without a performance
  conclusion.
- **D — invalid or abandoned attempt:** may explain a correction, but must not
  be counted as evidence.

A two-seed rollout-versus-current-leaf screen on two domains is defensible as
class B. If it is promising and used in a general conclusion, expand the chosen
two-arm comparison to ten matched seeds: 40 tasks total for two domains × ten
seeds × two methods. A neutral/futile pilot may stop at two seeds under a
predeclared gate and remain explicitly exploratory. Whether that evidence
level is acceptable for explaining a dropped direction is an explicit advisor
decision to obtain before submission; it is not silently assumed.

## Auxiliary training and selection methodology

| Method | Intuition | Evidence | Thesis status |
|---|---|---|---|
| Stage-1 ENHSP imitation | Retain the supervised planner-imitation procedure used to train the original Numeric ASNets. | The validation-selected Stage-1 policy is the baseline for every primary RQ. | Original-paper baseline, not a newly introduced thesis method. |
| Validation-selected checkpoints | Select a deployable model without test-set knowledge. | Primary reporting is validation-led only. MPrime required Phase B/C because the original validator saturated. | Main method; disclose MPrime correction. |
| Stage-2 MCTS-target refinement | Train on search visit distributions to improve beyond imitation. | No general Holm-significant policy-only gain; outcomes are domain-specific and one TPP seed collapsed. | Main mixed/negative result. |
| Constant Stage-1 KL anchor | Discourage destructive drift during Stage 2. | TPP revealed that the historical current-policy KL forward had dropout active. Across four optimizer RNG schedules, the selected catastrophic seed scored 11, 10, 20 and 19 under legacy semantics, but 20/20 in all four deterministic-current arms; the stable control stayed 20/20 throughout. | Main selected-seed implementation finding; stochastic susceptibility is demonstrated, not a population estimate. |
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
| Deterministic-current anchor KL | Keep replay training stochastic but compare deterministic current and Stage-1 anchor policies. | The catastrophic TPP seed is stochastically susceptible to legacy dropout-current KL, while deterministic-current KL protected it across all four tested optimizer RNG schedules. A frozen-replay crossover in the imperfect domains is dependency-submitted to isolate KL semantics from replay differences. | Selected-seed mechanism result; broader causal screen live. |
| MPrime validation redesign | Replace a saturated validator with harder frozen alternatives. | Phase B removed saturation; Phase C did not improve selector quality enough; Phase-B-A is final. | Methodological contribution. |

Until an explicit adoption decision is made, the primary RQs continue to
describe the historical implemented Stage-2 method, including dropout-current
KL. Corrected deterministic-current work is a causal/method-development branch.
If it is adopted as the new primary method, the affected primary Stage-2 cells
must be rerun consistently; old and corrected semantics must not be mixed
inside one claimed method family.

## Historical prototypes and evidence gaps

The early MCTS implementation used policy rollouts from leaves. Commit
`50835e09` introduced value-based leaf evaluation. This is distinct from the
Stage-1 teacher’s “ROLLOUT” experience mode. The git message says value-based
MCTS seemed slow, but there is no controlled rollout-versus-value comparison.
It may be described as an architectural transition, not as evidence that one
was better. A thesis-grade comparison would hold checkpoint, width,
simulations, PUCT and wall budget fixed and report coverage plus leaf-evaluation
runtime.

Other real code options lacking current matched evidence are not dismissed as
mere artifacts. They encode genuine hypotheses that were considered for
improving training or inference, and therefore form a ranked re-evaluation
portfolio:

The priority order uses four declared criteria: closeness to the primary
thesis/MCTS claim; ability to isolate one factor with matched evidence;
expected information per submitted job; and whether the method depends on a
prerequisite such as the value-head quality audit. It is a work-order ranking,
not a claim that lower-ranked ideas are scientifically invalid.

| Hypothesis family | Concrete arms | Scientific question | Priority / evidence plan |
|---|---|---|---|
| Leaf evaluation | rollout; learned-only; ENHSP-only; learned/ENHSP blend | Which leaf signal provides the best coverage/runtime trade-off? | Highest priority. Two domains x two seeds as a frozen exploratory screen; expand a promising main-claim arm to ten seeds. |
| Target construction | visit distribution; one-hot MCTS argmax (`8f69cc4b`) | Does preserving visit uncertainty help refinement more than a hard target? | High priority after leaf evaluation; matched frozen replay can isolate the loss. |
| Trajectory action policy | argmax; visit-proportional/policy sampling; epsilon/temperature schedules | Does stochastic action selection improve state coverage without destroying executable trajectories? | High priority, but freeze sampling schedules and RNG streams before comparison. |
| Replay augmentation | HER; goal-path reconstruction; extra tree states; visit-power sampling | Does relabeling or broader replay support improve learning from expensive searches? | Medium priority because it changes the data distribution and needs validity checks. |
| Estimator schedule | fixed blend; estimator decay (`779357ae`); heuristic bootstrapping; raw-value minimization (`84b2d60a`) | Should reliance on the external estimator decrease as the network learns? | Medium priority, preferably after the value-head quality audit. |
| Episode/reuse policy | skip/include failed searches; repeated optimization from one exploration | Are expensive failed or reused searches informative or biasing? | Lower-cost ablation once a primary training treatment is frozen. |
| Search safeguards/prototypes | duplicate penalties; corruption tests; removed PUCT variants (`0c5f6734`); parallel node generation | Do these repair a demonstrated mechanism or only add complexity? | Run only when tied to a concrete failure mode or runtime claim. |

These are separate from the MCTS inference families: fixed expansion,
progressive widening, and the leaf-evaluation family. The reason to revisit
them is the scientific hypothesis, not simply that code once existed. The
canonical prioritization, task counts and gates are in
`experiment_tracking/method_re_evaluation_roadmap_20260917.csv`.

For all re-evaluations, use the staged evidence rule: current-code compatibility
smoke; predeclared two-domain x two-seed matched screen; ten-seed expansion only
for a promising arm intended to support a broad outcome claim. A clearly
negative exploratory screen remains legitimate evidence for why a direction
was dropped, provided its stopping gate was frozen in advance.

## Held designs

- `(state, remaining horizon)` SAFE2: no demonstrated contamination and high
  memory risk.
- Path-batched multi-node PW: changes one-path/one-backup semantics.
- Action-count input ablation: requires fresh training.
- Original-paper stopping replication: compatibility experiment, not run.
- PUCT / estimator factorial: needed before claiming current `c=.1` and blend
  `.5` are optimal.
- Post-hoc PUCT sensitivity remains a documented next step if advisors request
  robustness evidence. It must be described as descriptive test sensitivity,
  not validation tuning or proof of a test-distribution optimum.
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
