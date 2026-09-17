# Imperfect-domain frozen-replay KL crossover

This is the causal repair of the first prospective KL-semantics screen.  The
same ten predeclared VH-off lineages are evaluated under legacy dropout-current
and deterministic-current KL, but each pair now consumes the exact same sixty
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

At the 17 September 2026 13:17 IDT snapshot, seventeen of twenty one-epoch
treatments were scheduler-complete and three Rover treatments were running.
The pair verifier and twenty endpoints remain dependency-gated, so training-log
validation scores are operational diagnostics only, not the declared result.
