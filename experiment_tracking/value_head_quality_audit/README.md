# Value-head quality side experiment

## Goal

RQ3 tests whether enabling the value head improves Stage-2 policy refinement;
RQ4 tests whether it changes the benefit of MCTS. Neither directly asks
whether the learned scalar value is calibrated or ranks successor states well.
MCTS currently blends learned value and an ENHSP estimate, normally at 0.5, so
MCTS coverage cannot isolate learned-value quality.

## Phase V1: offline calibration and sibling ranking

Use VH-on Stage-1 and validation-led Stage-2 checkpoints on the same frozen
validation states. Enumerate applicable successors in a batch and record raw
learned values. Keep four label sources separate:

1. held-out MCTS/replay target — target consistency, but partly circular;
2. deterministic continuation outcome and remaining length — practical utility;
3. raw ENHSP `h` — independent external planner evidence for rank/regret, not
   guaranteed optimal value;
4. `enhsp_search_v = exp(-1.0 * h)` — the exact bounded transform used by the
   deployed non-minimization search, retained as a separate scale-aligned
   diagnostic with a checksumed transform configuration.

Metrics are label-semantic-specific. MAE/MSE and calibration bins are allowed
only for a label explicitly marked scale-comparable: exact persisted replay
`z` and the separately named `enhsp_search_v`. Raw ENHSP `h` and raw remaining-action counts are not on the learned
bounded/transformed value scale, so they support Spearman/Kendall sibling
ranking, pairwise ordering, top-child regret and best-successor agreement only.
Goal, unsolved and timeout outcomes support separate classification metrics;
they are not numeric calibration targets. Also report matched Stage-1 to
Stage-2 change per seed/domain rather than pretending every state is an
independent replicate.

The original recommended pilot was Rover, Drone and FO Counters with two
predeclared VH-on seeds per domain. MPrime's exact Phase-B-A Stage-1 and final
validation-led Stage-2 endpoint paths are now recorded locally, and both
Stage-2 hashes are present. MPrime is therefore included as a conditional
fourth domain. The design now contains **16 checkpoint tasks**, not 12. It
remains blocked until every checkpoint and state manifest has a verified hash.
Avoid Block Grouping initially because successor generation is expensive and
ordinary Counters because extremely long trajectories can dominate.

### Implemented local preparation (2026-09-17)

`asnets/value_head_audit.py` now provides an opt-in, side-effect-free primitive
that enumerates every applicable successor, evaluates raw learned values in one
network batch, and preserves action/successor identity, transition probability,
and terminal/goal flags. Production training, policy inference, and MCTS do not
import it. Unit tests cover batching, stochastic-successor identity, invalid
probability mass, and rejection of VH-off networks.

`v1_manifest_template.csv` freezes task/checkpoint/state-manifest provenance.
`v1_successor_row_schema.csv` separates raw network outputs from label source,
status, orientation, comparability, and log provenance. This prevents replay,
continuation, and ENHSP targets from being silently pooled.

`v1_checkpoint_candidates.csv` now freezes seeds `534933607` and `923500475`
without selecting them on audit outcomes. For Drone, FO Counters and Rover it
uses the canonical MAIN-VAL validation-selected Stage-1 and Stage-2 rows. For
MPrime it uses the Phase-B-A Stage-1 selector and final validation-led Stage-2
manifest. The candidate builder asserts the complete factorial and records a
checksum of each source ledger.

`v1_checkpoint_hashes.csv` records the exact `weights.joblib` SHA-256 for all
sixteen endpoints. `v1_transform_config_audit.csv` records an explicit log
match for all sixteen runs: coefficient 1.0 and non-minimization mode.

`v1_state_mixture.csv` predeclares one 60-state manifest per domain/seed
lineage: 20 common planner-reached states, 20 common seeded-random legal-walk
states, 10 Stage-1-policy states, and 10 Stage-2-policy states. Both checkpoints
in a lineage must reference the exact same manifest and hash. The common 40
states provide the primary paired comparison. Results on the two policy-state
strata are reported separately as distribution-sensitivity evidence: a state
from one policy is an intentionally out-of-distribution stress case for the
other, not a claim about its natural trajectory distribution. No network is
updated from these states, so this is a paired offline evaluation, not shared
replay training.

`v1_label_sources.csv` freezes label meanings, orientations, scale
comparability, cache identities, timeouts and prohibited fallbacks. Replay `z`
and `enhsp_search_v` are comparable bounded scales, while the latter remains
planner/search evidence rather than optimal ground truth. Raw ENHSP `h` and raw
continuation length are explicitly non-comparable and rank/regret-only. Local
provider scaffolding enforces that a timeout/unsolved/error cannot carry a
fabricated numeric label. A strict preflight checks these comparability rules,
the 16-task factorial, hashes, resources, label families, and the shared
Stage-1/Stage-2 state-manifest identity.

The post-capture chain is implemented but not scientifically released. Its
task runner verifies checkpoint, paired-manifest, optional replay-cache, and
ENHSP-transform hashes before loading a state, then writes the four label
families as separate status-bearing rows. Eight six-state resource tasks (both
stages for one frozen seed per domain) must show zero provider errors and at
least 20% measured memory/runtime headroom. Only a passing checksumed resource
gate permits the release controller to submit the four domain arrays comprising
the sixteen scientific tasks. The dependent output finalizer verifies task and
file completeness but deliberately makes no RQ claim before paired aggregation.

This is still preparation, not an executed V1 scientific result, and no V1
scientific checkpoint task has been submitted. All sixteen checkpoint payloads
have now been verified and recorded in `v1_checkpoint_hashes.csv` (fourteen
newly materialized hashes plus the two already recorded MPrime Stage-2 hashes).
The eight shared state manifests remain to be materialized, so submission
readiness is not yet claimed.

On 18 September 2026, smoke `21453653` correctly exposed that the isolated
checkout lacked the ignored compiled TensorFlow operator; no scientific work
ran. The checksum-verified production operator was linked into the isolated
checkout. Smoke `21453918` then reached state capture but exposed that
`to_tup_state()` omits MDPSim special fluents, so its round-trip record could
not be restored. Commit `2b86060b` fixes the serializer to preserve the full
fluent vector, including total cost, and passes 17/17 focused tests. Repaired
smoke `21454176` is resource-pending. The four two-task capture arrays are
dependency-pending as `21454177` (Drone), `21454178` (FO Counters), `21454179`
(Rover), and `21454180` (MPrime), with finalizer `21454181` dependent on all
eight captures. No user input remains: the paired-state manifests materialize
automatically after the smoke passes.

### V1 execution plan and resources

1. Preserve the completed transform audit: all sixteen logs explicitly record
   coefficient 1.0 and non-minimization mode, matching `exp(-1.0 * h)`.
2. Materialize the eight checksumed 60-state lineage manifests and cache
   successor identities once. Stage-1 and Stage-2 must reference the same file.
   Cache each label family separately; never turn a timeout from one labeler
   into a numeric target from another.
3. Run a six-state/domain preflight (two common planner, two common random, one
   state from each policy stratum). Measure peak RSS, successors/state, ENHSP
   latency, continuation latency and missing replay-label rate before releasing
   the pilot.
4. Run the sixteen checkpoint tasks, then aggregate at the lineage/domain level.
   State-level observations within a trajectory are not treated as independent
   seeds.
5. Accept the V1 gate only if sibling ordering, top-child regret and dead-end
   discrimination are reproducible across both seeds in at least two domains.
   Global correlation alone is insufficient.

The provisional per-task requests are Drone 4 CPU/48 GiB/8 h, FO Counters
4 CPU/64 GiB/8 h, Rover 4 CPU/96 GiB/12 h, and MPrime 4 CPU/120 GiB/12 h.
Releasing all 16 simultaneously would request 64 CPU and 1,312 GiB. These are
ceilings pending the six-state preflight, not measured requirements. Label
cache construction is a separate workload and is not included in that total.
Every aggregate CSV must include checkpoint, state-manifest and raw label-log
paths and hashes.

### Remaining ambiguities and release blockers

- The checkpoint payload gate is complete: all sixteen hashes are frozen, and
  source-ledger selection remains independent of audit outcomes.
- The eight state manifests and their canonical serialized-state hashes do not
  exist yet; the plan fixes quotas and selection rules, not the realized states.
- Historical replay targets may be absent for validation states. Missing is an
  allowed observation; generating new checkpoint-specific MCTS targets would
  be a separate, explicitly circular diagnostic and is not silently substituted.
- Deterministic continuation uses the audited checkpoint's stable policy
  argmax. It is therefore a within-checkpoint utility label, not common ground
  truth for the paired Stage-1/Stage-2 comparison.
- ENHSP raw heuristic values are independent planner evidence, not optimal
  costs. State injection, planner status parsing and cache writing still need
  the domain-specific adapter and smoke test.
- Raw ENHSP `h` and remaining-action counts remain non-comparable. V1 now adds
  the separately named `enhsp_search_v` using the exact current-search formula,
  coefficient/minimization mode and checksumed transform config. It is not the
  reciprocal helper used elsewhere for training-target construction and is not
  labeled optimal ground truth.
- The saved configuration must confirm raw-value orientation, minimization mode
  and target transform before any scale-sensitive metric is computed; the
  schema records orientation and scale comparability explicitly.
- Resource requests and the 300 s ENHSP / 900 s continuation limits are
  provisional until the four-domain preflight measures tails and peak memory.
- MPrime expands the pilot from 12 to 16 tasks. Both Stage-1 and both Stage-2
  payload hashes are frozen; its remaining gate is the same paired-state
  materialization and measured-resource preflight as the other domains.

## Phase V2: raw-value-greedy inference

Only proceed if V1 shows useful sibling ranking. At each external step:

1. enumerate applicable actions and batch-generate successors;
2. evaluate raw learned value only;
3. mask known terminal non-goals and prioritize an immediate goal;
4. choose maximum value, using stable action ID only for a remaining exact tie.

Compare the same checkpoint under policy argmax, raw-value greedy, optionally
ENHSP-only greedy, and optionally the existing 0.5 blend. Report coverage,
action count, successors generated, runtime and failure mode. This diagnostic
is feasible because batched successor generation and network inference already
exist; implementation plus tests is estimated at one to two development days.
The validation screen would add 12 evaluation tasks at roughly 2 CPU / 120 GiB
each.

### V2 execution plan and resources

Implement the new inference mode behind an explicit flag, leaving policy and
MCTS behavior unchanged.  Unit-test batching, goal/terminal precedence,
numeric minimization/maximization sign, successor/action identity and stable
ties.  Smoke-test one short instance before releasing the twelve validation
tasks.

Development plus tests is estimated at one to two working days.  The first
screen requests at most 24 CPU / 1.44 TiB concurrently and uses a 6–12 hour
task bound after a short runtime preflight.  It is followed by policy-argmax
and value-greedy comparisons on the same checkpoints and instances; no MCTS
run is needed to answer the narrow diagnostic.

V2 is normally gated by V1.  The gate may be deliberately overridden only by
freezing that decision before viewing V2 test outcomes and labelling the run
as an exploratory inference diagnostic.  Skipping a failed V1 gate cannot be
used to claim that the value head is calibrated; at most it can reveal a
surprising trajectory-level behavior worth investigating.

## Interpretation limits

Planner estimates are not optimal ground truth; label sources must not be
pooled. Raw-cost labels must not be used for MAE/MSE or calibration against the
bounded learned value. Global correlations can be driven by instance
difficulty, so sibling ranking is more informative. Policy-only states have
survivorship bias; sample off-policy/planner-reachable states too. A failed
value-greedy trajectory can reflect compounding distribution shift or
successor-generation cost as well as poor calibration.

## Repaired execution chain — 18 September 2026, 19:37 IDT

Smoke `21455175` passed after commit `7ebc2c16` restored action-history inputs
using valid initial-state generator context and then reinstated the exact
captured auxiliary vector. Capture arrays `21455176`–`21455179` are running
for Drone, FO Counters, Rover and MPrime. Finalizer `21455180` and post-capture
controller `21455181` are dependency-pending. If the captures pass, the latter
automatically launches the measured resource preflight, its checksum gate, the
16 scientific checkpoint tasks and the scientific finalizer. No further user
action is required.

## Overnight capture audit and exact recovery — 19 September 2026

The eight capture tasks did not complete, and the previous finalizer/controller
were cancelled by failed dependencies.  The durable outputs establish three
separate causes:

- Drone completed all four capture sources, but manifest materialization
  rejected an exact state hash that appeared in more than one frozen stratum.
- Both FO Counters tasks exhausted the 64-GiB allocation while buffering long
  trajectories in worker memory.
- Rover and MPrime stopped partway through their later sources.  MPrime's
  random actors had already buffered 90,000--100,000 state records per lineage;
  Rover's incomplete arm similarly stopped during Stage-2 policy capture.

This is an execution failure, not a V1 scientific result.  Completed source
files remain reusable.  Commit `29deb4b3` implements the minimal repair:

1. V1 capture workers keep the eight lowest-ranked unique candidates per
   instance instead of buffering an entire trajectory.  This is opt-in and
   does not change ordinary policy evaluation, training, or MCTS.
2. Recovery jobs can name only missing sources; completed sources are not
   rerun.
3. Manifest selection fills each already frozen source quota in balanced rounds
   across the predeclared instance pool.  An actor timeout is therefore treated
   as missing actor-reachable state evidence, not as permission to choose a new
   instance after seeing value-head outputs.  Exact duplicates fall through to
   the next deterministic candidate.
4. The manifest sidecar records how many distinct frozen instances represent
   each source, exposing any feasibility-driven concentration.

The exact recovery plan contains six source-capture tasks: two FO random
captures, one FO Stage-2 policy capture, one Rover Stage-2 policy capture, and
two MPrime random-plus-policy captures.  Their maximum simultaneous request is
30 worker CPUs and 624 GiB.  Only after all six succeed does an eight-task
manifest rebuild run, followed by the existing strict finalizer, measured
resource preflight and 16-task scientific V1 release.  A real-data Drone smoke
must pass before the recovery controller releases that chain.

At the 19 September preflight, the shared cluster home quota first prevented
the last smoke/controller files from being written to the isolated checkout.
No scientific captures were deleted to work around it.  Removing only
reproducible Python bytecode caches created enough space for the two committed,
checksum-verified scripts.  Smoke `21463884` is running with 1 CPU / 8 GiB;
controller `21463885` is `afterok:21463884` and requests 1 CPU / 2 GiB.  The
controller cannot release recovery or downstream scientific work unless the
real-data Drone smoke succeeds.

Smoke `21463884` completed in 73 seconds: all twelve existing audit tests and
both new manifest-selection tests passed, and both durable Drone capture trees
materialized valid 60-state temporary manifests.  Controller `21463885` then
released the canonical exact chain:

- FO random recovery `21463887_[0-1]`: 6 allocated CPUs, 96 GiB and four hours
  per task;
- FO seed-923500475 Stage-2 policy recovery `21463888_1`: 6 CPUs, 96 GiB and
  four hours;
- Rover seed-534933607 Stage-2 policy recovery `21463889_0`: 6 CPUs, 96 GiB and
  four hours;
- MPrime random plus both policy sources `21463890_[0-1]`: 6 CPUs, 120 GiB and
  eight hours per task;
- eight-manifest rebuild `21463891_[0-7]`, strict state finalizer `21463892`,
  and automatic preflight/science controller `21463893`, all linked by
  `afterok` dependencies.

All six source-recovery tasks began running immediately.  Their maximum active
request is 36 allocated CPUs and 624 GiB.  Existing complete source captures
remain in place and are joined by the deterministic manifest rebuild; they are
not recomputed by these filtered jobs.

### Operational recovery revision — 19 September 2026, 12:00 IDT

The initial released jobs did not all reach the V1 program.  FO-random array
element `21463887_0` (actual job `21463894`) failed after 45 seconds with only
the scheduler prologue in its log.  Its sibling and the FO/Rover recoveries
were requeued and held with `user env retrieval failed`; both MPrime elements
failed before producing program output.  These are operational launch
failures, not failed captures or scientific outcomes.  Consequently the
original downstream chain `21463891`--`21463893` and two intermediate
replacement chains were dependency-cancelled and are obsolete.

The shared home quota was again unable to create even the tiny recovery
ledger.  Only regenerated `__pycache__` directories in the isolated V1
checkout were removed; no capture, checkpoint, log or scientific result was
deleted.  A write/delete probe then succeeded.  The exact held recoveries were
released and exact MPrime replacements were submitted while preserving every
durable source capture.  All six required recovery identities were verified as
genuinely running after the cleanup:

- FO random seed 534933607: `21463904_0`, 96 GiB / four hours;
- FO random seed 923500475: `21463920_1`, 96 GiB / four hours;
- FO Stage-2 policy seed 923500475: `21463888_1`, 96 GiB / four hours;
- Rover Stage-2 policy seed 534933607: `21463889_0`, 96 GiB / four hours;
- MPrime random plus both policy sources: `21463944_[0-1]`, 120 GiB / eight
  hours per seed.

The canonical downstream chain is now `21463956_[0-7]` (deterministic
manifest rebuild) -> `21463957` (strict finalizer) -> `21463969` (automatic
resource preflight, checksum gate and scientific release).  Its dependency
record contains only the live exact recoveries above.  The maximum concurrent
capture request remains 36 allocated CPUs and 624 GiB; the four- and eight-hour
figures are scheduler bounds, while the bounded-capture implementation is
expected to finish substantially sooner if the cluster launch environment
remains healthy.
