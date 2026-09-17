# Imperfect-domain frozen-replay KL crossover

This is the causal repair of the first prospective KL-semantics screen.  Ten
predeclared optimizer/replay schedules across six source checkpoints are
evaluated under legacy dropout-current and deterministic-current KL. Each pair consumes the exact same sixty
captured replay batches and targets from the corresponding legacy arm.  Both
arms also share the starting checkpoint and fixed optimizer-step RNG schedule.
Only the current-policy forward used inside the anchor KL differs.

The frozen source is deliberately the already captured `now_legacy` schedule;
the choice is made before viewing crossover results. Before submission, SHA-256
values for every one of the 600 replay files and all ten source checkpoints are
frozen in pair-specific manifests. Every task fails closed unless its manifest
verifies, all sixty source batch files exist, every source SHA-256 is recorded,
the fixed step RNG is correct and the expected KL forward is logged. A dependent
pair verifier then requires the ordered replay hashes, step seeds, pre-treatment
policy gradient and target-disagreement bits to agree before endpoint scoring.

Scope: twenty one-epoch training tasks plus twenty policy endpoints. Here one
epoch is exactly sixty frozen optimizer steps. The jobs already bypass fresh
MCTS exploration and target generation; they apply the sixty updates and then
run the normal harness's inline validation/snapshot step. Applying those
gradients *is* the remaining scientific training treatment. A future
replay-only harness could remove grounding, inline validation and normal
trainer/container/checkpoint setup overhead, but it cannot remove the optimizer
steps or declared endpoint evaluation and would need its own equivalence smoke
test. Training requests 6 CPU / 120 GiB / 4h per task;
endpoints request 5 CPU / 20 GiB / 2h.
The observed non-frozen one-epoch runs took about three minutes to 2h03, while
frozen replay skips target generation and should normally be faster.  Slurm is
allowed to schedule this alongside or before MPrime without manual priority
changes.

Primary comparison: frozen deterministic-current minus frozen legacy policy
coverage within each exact lineage. Historical epoch 0 and independently
generated prospective endpoints remain context only.  No 100-epoch retraining
is implied unless this crossover demonstrates repeatable harm.

The Preserve-3 decision is deliberately gated. Delivery and Zenotravel are
already at or near policy-coverage ceiling, while the demonstrated
susceptibility is a selected TPP seed. We will not blanket-rerun all Preserve-3
Stage-2 lineages from this one mechanism result. If the imperfect-domain
crossover shows common harm, or if deterministic-current KL is adopted as the
new primary Stage-2 method, the scientifically consistent follow-up is a
predeclared corrected-KL campaign (first TPP matched seeds, then all primary
cells needed for a uniform corrected-method RQ). We must not selectively mix
legacy and corrected semantics inside one primary comparison.

At the 17 September 2026 15:28 IDT snapshot, all twenty one-epoch treatments
were scheduler-complete and verifier `21430552` had certified all ten pairs:
the ordered 60 replay hashes, step-0 policy gradient and step-0 target-argmax
bits matched within every pair. Nineteen of twenty declared test endpoints were
complete. The remaining Block Grouping legacy endpoint, array task 0, is held
after Slurm failed to retrieve the user environment; it has not begun inference.

The completed paired endpoint effects are heterogeneous:

| Domain / optimizer schedule | Exact source policy | Legacy | Deterministic-current | Change |
|---|---:|---:|---:|---:|
| Block Grouping / 42 | 17/20 | 15/20 | 15/20 | 0 |
| Block Grouping / 2026 | same source as row above | 13/20 | 15/20 | +2 |
| Drone / 2026 | pending exact evaluation | 5/20 | 5/20 | 0 |
| Drone / 42 | same source as row above | 6/20 | 4/20 | -2 |
| FO Counters / 2026 | pending exact evaluation | 6/20 | 3/20 | -3 |
| FO Counters / 42 | same source as row above | 7/20 | 4/20 | -3 |
| Rover / 42 | pending exact evaluation | 4/20 | 4/20 | 0 |
| Rover / 2026 | same source as row above | 4/20 | 4/20 | 0 |
| Counters / 534933607 | 59/59 | 6/59 | 35/59 | +29 |
| Counters / 2082152039 | 35/59 | 59/59 | 38/59 | -21 |

The `seed` labels in this compact screen are optimizer/replay RNG schedules,
not ten independent primary network lineages.  Block Grouping, Drone, FO
Counters and Rover each use one source checkpoint under two RNG schedules;
Counters uses two distinct source checkpoints.  The screen therefore proves a
mechanism and exposes susceptibility, but it is not a two-network-per-domain
population sample.

The causal mechanism is clear but the performance direction is not uniform.
At identical starting weights the deterministic-current anchor gradient is
approximately zero, while legacy dropout-current produces a substantial
artificial anchor gradient in every domain. Removing that gradient can help,
hurt or leave one-epoch policy coverage unchanged depending on the lineage.
Across the nine complete optimizer schedules the median coverage change is zero; the extreme
opposite Counters effects show that a blanket corrected-KL retraining decision
is not yet justified. The different Counters starting networks and replay/RNG
schedules can plausibly produce opposite treatment effects, so this screen does
not isolate network identity as the moderator. No primary RQ changes from this
selected mechanism screen.
Row-level endpoints and log pointers are in
`partial_endpoint_results_20260917_1528.csv`; gradient evidence is in
`first_step_gradient_results_20260917_1528.csv`.

At 16:26 IDT the exact held task was released.  It started as component job
`21433102` on `ise-cpu-intl-12` and failed before test inference with native
exit code -4 after 87 seconds.  This is operational failure, not a policy
score.  The failed log is preserved.  Exact replacement array `21434122_[0]`
uses the same checkpoint, evaluator and 5-CPU/20-GiB/two-hour request, writes a
new output file, and excludes the failed node in addition to the campaign's
existing exclusions. It completed on `ise-cpu-intl-19` at 15/20, making this
paired effect exactly zero.

## Frozen next decision gate

Do **not** jump directly from this selected-checkpoint screen to a five-domain,
100-epoch replacement campaign.  The next defensible step is a primary-lineage
one-epoch susceptibility screen:

1. use each domain's ten validation-selected Stage-1 VH-off checkpoints;
2. capture one exact 60-batch replay schedule per lineage;
3. compare legacy and deterministic-current KL on that identical checkpoint,
   replay, targets and optimizer RNG;
4. evaluate both one-epoch endpoints with policy inference; and
5. estimate how often and how severely immediate coverage changes across
   independent network lineages.

Exact provenance currently supports reuse of the two Counters primary
lineages.  The older single checkpoints used for Block Grouping, Drone, FO
Counters and Rover have not been proven hash-identical to current MAIN-VAL
primary lineages and therefore are mechanism evidence only. A uniform
ten-lineage screen therefore needs 48 additional primary checkpoint lineages,
96 one-epoch treatments and 96 endpoints, not “eight more seeds” from every
current row. That screen is now submitted as `21434692`–`21434696`, without an
artificial concurrency throttle: Slurm admits tasks within the 6-TiB running
limit and leaves excess work pending. Exact source-policy baselines for the
four older shared checkpoints are `21434691` plus exact retry `21434754`;
until those finish, their source-score cells remain explicitly pending.

One epoch answers prevalence of the initial-update effect; it cannot answer
final 100-epoch performance.  After that screen, choose one stable/cheap domain
and one high-effect or high-variance domain for matched ten-lineage, 100-epoch
legacy-versus-deterministic training, complete policy curves and endpoints,
then matched fixed/PW search only if policy evidence makes it relevant.  This
separates the semantic correction question from the much more expensive
decision to replace the primary training method.
