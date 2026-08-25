# Experiment pipeline: deployment boundary and next steps

Last updated: 2026-08-25.

## Production-code boundary

The production cluster checkout remains pinned at `b6985638` while the current
training and policy-evaluation queue drains. Existing running processes would
not change after a pull, but queued jobs resolve the checkout only when they
start. Pulling now would therefore mix code revisions inside the same
replication campaign.

The following tested work is available on
`codex/thesis-reproducibility-bundle` but is deliberately not deployed yet:

- `5822426b`: reap timed-out evaluation workers before replacing their pool
  slot; prevent abandoned JVM/tree workers; fail clearly if the worker cannot
  be reaped.
- `fac11cb0`: opt-in progressive widening and remaining-horizon-aware MCTS
  inference. Both features default off and are separate from the lifecycle
  fix.

Do not release any MCTS controller before the production deployment below.

## Held-MCTS scheduling action

On 2026-08-25, 181 held MCTS evaluations and the one held MCTS controller were
cancelled to free Slurm's submitted-job limit for training and policy work.
They consumed no compute resources while held but did count toward the user
job-count ceiling.

The exact preserved records are on the cluster at:

- `2026-08-22/statistical_replication_controller_option_a/held_live.tsv`
- `2026-08-22/statistical_replication_controller_option_a/mcts_cancelled_requeue_20260825.tsv`
- `2026-08-22/statistical_replication_controller_option_a/pending_mcts_held_20260824T092957.txt`

The cancelled job IDs must not be treated as completed scientific results.
Their manifest configurations remain the source for later idempotent
resubmission.

## Deployment sequence after non-MCTS work drains

1. Confirm that the current training and policy-evaluation jobs have drained.
2. Merge the tested feature branch into `main`.
3. Pull the resulting commit into the production cluster checkout.
4. Run the rolling-evaluation and JPDDL lifecycle smoke tests in the production
   Apptainer environment.
5. Verify the cluster checkout commit in every new submission log.
6. Resume affected FO Counters and Rover MCTS evaluations with two workers and
   160 GB, seeding their existing completed-instance manifests.
7. Resubmit the corrected Block Grouping VH-off width-5, 20-iteration jobs.
8. Release the remaining MCTS manifests through a dedicated MCTS-only
   controller.

The two-worker/160-GB deviation is a resource-scheduling adjustment, not an
algorithmic change. It reduces simultaneous retained-tree/JVM memory while
preserving the checkpoint, search parameters, per-instance time limit and
instance order. Reports must record this resource adjustment explicitly.

## Separate held experiments

- PUCT sensitivity remains held.
- The Drone progressive-widening pilot remains held until the lifecycle code is
  deployed. It compares fixed top-20, fixed top-5, and two progressive variants
  at 70 simulations on matched checkpoints.
- Remaining-horizon enforcement is a separate matched experiment and must not
  initially be combined with progressive widening.
- Horizon-indexed `(state, remaining horizon)` statistics remain a documented
  deferred design because of their potential memory cost.

## Four-domain Stage-2 anchor selection

Delivery, MPrime, TPP and Zenotravel use the same seven-value Stage-2 anchor
grid used for the earlier five-domain campaign:

`0, 0.03, 0.3, 1, 3, 10, 30`

The tuning lineages reuse Stage-1 seeds `1963100312` and `2011206605` from the
current 80-lineage campaign. These two seeds are the tuning set; the remaining
eight seeds per domain/value-head cell are the held-out confirmation set. This
avoids 16 redundant Stage-1 training jobs.

Using these two seeds instead of the historical labels `42` and `2026` does
not change the statistical design: the important requirement is that the
tuning seeds are declared before examining Stage-2 test outcomes and are not
later counted as independent held-out confirmation seeds. The seed values
themselves have no special scientific meaning.

The complete tuning workload is 112 Stage-2 training jobs
(`4 domains x 2 value-head modes x 2 tuning seeds x 7 coefficients`) followed
by 2,352 every-five policy evaluations. Select the coefficient using the
predeclared two-seed mean learning-curve AUC, with mean peak and final coverage
as tie-breakers. Only after coefficient selection should the remaining eight
held-out seeds enter the confirmatory Stage-2 pipeline.
