# Counters Stage-1 VH-off strict tie-break confirmation

This is the predeclared domain-wide confirmation of the targeted three-instance
causal result. It compares only two rules on the same current build:

- `action_id`: the historical lowest-global-action-index resolution of equal
  maximum root visit counts;
- `policy`: among equal maximum visits, use the largest network prior, then the
  stable action ID only for a remaining exact tie.

All ten validation-selected Stage-1 VH-off checkpoints and all 59 Counters test
instances are rerun under both rules. Every other setting is matched: narrow
search (five children, 20 simulations), PUCT 0.1, estimator mixture 0.5, three
workers, six-hour per-instance timeout, 10,000 external-action cap, six CPUs,
120 GiB and a 72-hour allocation. The array has low Slurm priority
(`nice=10000`). Q-only tie-breaking is excluded because it rescued 0/3 targeted
failures and is not a plausible default candidate.

The scientific scope is 20 full evaluation tasks (10 seeds x 2 rules). This
answers whether the policy tie-break improves *domain-wide* coverage, rather
than only the deliberately selected failures. It also makes the baseline and
candidate strictly comparable under the same build and scheduler-era setup.

`manifest.csv` carries the exact source checkpoint, Stage-1 training job,
policy-evaluation job, policy score and eventual remote output directory for
every task.

## Live status

At the 14 September 2026 15:02 IDT snapshot, six tasks were terminal and 14
were running. Three complete matched seed pairs were neutral: seed
`1073581256` scored 59/59 under both rules, seed `1239739722` scored 17/59
under both, and seed `2011206605` scored 20/59 under both. The matched
three-pair effect is exactly zero, but these three checkpoints contain zero
classified policy-success/action-ID-failure opportunities. Their neutrality
therefore does not test the proposed rescue mechanism. The informative
high-policy-coverage pairs remain live; unpaired lower bounds must not be used
as a treatment comparison. RQ2 remains frozen until all ten pairs terminate.

The complete interim, seed-level join is in `progress_20260914_1502.csv`. It
records the exact validation-selected checkpoint, source training and policy
logs, pure-policy score, classified count, search successes, classified
policy-success/action-ID-failure opportunities, gains over policy, and
unclassified policy-success instances for both rules. Current workload
provenance is refreshed separately in the advisor-followup workload ledger.

At 14 September 23:19 IDT, seven tasks were terminal and thirteen were still
running. The same three matched pairs above remained the only complete pairs,
so the domain-wide treatment estimate and RQ2 table remain frozen. Current
durable ledgers contain 255 action-ID successes among 467 classified outcomes
and 253 policy-prior successes among 475 classified outcomes; these unpaired
lower bounds are operational progress only and are not a treatment comparison.

## Instrumentation limit and follow-up gate

The strict campaign was designed as a full-domain coverage confirmation. It
used lean worker logs and deliberately omitted per-root `--action-debug` to
avoid enormous I/O over 1,180 seed-instance evaluations. It was therefore a
mistake to expect every first-divergence root vector to be recoverable from
these logs afterward. The separate three-instance pilot did record the root
vectors and remains the valid mechanism evidence.

After all twenty strict tasks classify, the minimal mechanism repair is to
rerun only the union of policy-success/action-ID-failure identities with
`--action-debug` under the exact checkpoint, build and search configuration.
Within-search `--puct-debug` is optional and should be added only if the root
vector cannot explain the choice, because it is substantially more verbose.
No successful instance or full-domain arm needs to be repeated.

A possible future *search-abstention* treatment is policy-relative rather than
time-relative: accept the MCTS winner only when it leads the applicable policy
argmax by at least two visits; otherwise follow the policy. A two-visit margin
cannot be reversed by moving a single simulation. A stricter alternative also
requires a majority (at least 11 of 20 edge visits), but may discard valid
multi-action search evidence. Neither threshold is established by the current
test failures; any candidate must be calibrated and frozen on validation
traces before a test-set comparison.

## Dependency-gated full-root trace follow-up (submitted 15 September)

The exact mechanism follow-up is implemented. The original controller
`21319149` was cancelled and superseded after the timeout-ledger defect below
was found.
The first corrected controller `21347597` was superseded after a later OOM in
strict task 9 exposed additional unclassified identities. Current controller
`21362904` waits for strict array `21233925`, recovery `21347260`, and exact
task-9 recovery `21362903`, then
fails closed unless all ten action-ID ledgers contain exactly one terminal
record for every evaluator identity 1-59 and each parsed policy score matches
the frozen manifest. Inclusion is exactly `policy success AND same-build
action-ID MCTS failure`. Each included seed-instance is emitted exactly twice,
under `action_id` and `policy`; policy failures, MCTS successes, duplicates and
unclassified records cannot enter.

Each trace task retains the exact Stage-1 checkpoint and narrow 5/20 settings,
runs one identity with one worker, two CPUs, 120 GiB, a six-hour instance limit
and an eight-hour walltime, and records full `--action-debug` root vectors.
`--puct-debug` remains disabled unless a root vector later proves insufficient.
The low-priority array is capped at 12 tasks, hence at most 24 CPUs and 1,440
GiB. If the final strict evidence contains `N` wanted identities, the array has
exactly `2N` tasks and a queue-excluded wave bound of
`ceil(2N / 12) * 8 hours`.

The generated `full_trace_manifest.csv` and `full_trace_source_summary.csv`
retain the source training jobs, policy jobs, checkpoints, logs and completion
ledgers. The controller uses `afterany` because OOM-labelled strict tasks are
completed by a separate recovery array; the manifest builder, not the Slurm
label, proves scientific completeness. Submission is pinned to the isolated
checkout through core-code commit
`de1d29b83c6900e15de772074f8dd5595803dbbd`. The controller is explicitly
non-requeueable and refuses to submit a duplicate trace array if its durable
submission record already exists. Deployed script hashes, exact dependencies,
resources and paths are recorded in `full_trace_submission_20260915.csv`; the
controller will additionally freeze the actual inputs in
`deployed_inputs.sha256` beside its scheduler output.

## 15-16 September correction: timeout evidence and exact recovery

The first recovery design was not actually exact. The rolling evaluator wrote
success and ordinary 10,000-action outcomes to JSONL but printed six-hour hard
timeouts only to stdout. Consequently recovery `21308619[4,6,14]` treated
already classified timeouts as absent and began repeating them, while the first
controller could never satisfy its 59-JSONL-row gate.

The three recovery tasks were checksummed, cancelled without deleting output,
and reconciled using only explicit
`[EVAL INSTANCE] timeout ... limit=21600.0s` events. No absence or Slurm walltime
was interpreted as a scientific timeout. This proved that tasks 4, 6 and 14
were already scientifically complete. The same reconciliation completed tasks
16 and 17. At that snapshot only three identities remained genuinely
unclassified: task 7 had one crashed identity and task 19 had two. A later
task-9 OOM occurred after the snapshot. Reconciliation over its new terminal
log proved 56/59 terminal identities; evaluator identities 45, 48 and 52 were
genuinely unclassified.

Exact replacement `21347260[7,19]` runs only the original three identities,
with one worker per task, two CPUs, 120 GiB, a six-hour per-instance cap and a
14-hour walltime. Exact task-9 replacement `21362903[9]` runs only identities
45, 48 and 52 with one worker, two CPUs, 120 GiB, the original six-hour
per-instance cap and a 20-hour walltime (`3 * 6h` plus two hours overhead).
Interrupted MCTS trees cannot resume within an instance: durable terminal
identities are skipped, while each interrupted identity restarts with its full
scientific six-hour cap. Pre-scientific setup failures and two native exit-4
attempts are preserved as provenance. Both native failures were on
`ise-cpu-intl-25`, which the current recovery excludes without excluding its
wider node family.

Current controller `21362904` depends on the original strict array and both
exact recovery groups. Before building the trace manifest it reconciles
all ten action-ID logs, requires exactly 59 unique terminal identities per seed,
allows only success, ordinary-unsolved and explicit hard-timeout statuses, and
then emits only `policy success AND action-ID failure` identities under the two
tie rules. The broader visit-margin calibration remains documented but held
until this exact-tie confirmation and trace follow-up finish.

At the 16 September 2026 01:37 IDT refresh, task 7's sole missing identity
(`fz_instance_56.pddl`) reached the exact 21,600-second limit and printed an
explicit timeout event. It is scientifically terminal. Its JSONL remains at
58/59 only because the recovery runner reconciles before evaluation; controller
`21362904` performs the required post-terminal reconciliation before applying
the 59/59 gate, so the identity will not be rerun. Task 19 also printed an
explicit timeout for its first of two missing identities and is evaluating the
second. Task 9 is evaluating the first of three exact identities. No RQ2 score
changes until all ten matched pairs and the reconciliation gate are complete.
