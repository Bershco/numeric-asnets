# All-imperfect-domain rollout leaf-evaluator screen

## Scope and nonduplication

This is the thesis-wide, evidence-class-B characterization of the dropped
policy-rollout leaf evaluator. It is one analysis design assembled from two
nonoverlapping launch manifests:

- the already submitted 16-task Drone + FO Counters four-arm subset in
  `experiment_tracking/mcts_leaf_evaluator_four_arm_20260920/manifest.csv`;
- the 16-task extension in this directory's `manifest.csv`.

The complete analysis therefore has 32 science identities, not 48:

- Block Grouping, Drone, FO Counters, Rover, Counters and MPrime;
- two predeclared Stage-1 VH-on seeds per domain (`1963100312` and
  `2011206605`);
- rollout versus the current `0.5` learned/ENHSP blend in all six domains;
- learned-only and ENHSP-only only in the existing Drone/FO subset.

The extension must not rerun any Drone or FO Counters identity. The existing
four-arm science array is Slurm job `21483378_[0-15]`, launched from code
commit `2b243a6e8f21723f99cfd84136770ad32d220ffa`. Its rollout and blend rows are
the primary all-domain comparison for those two domains.

Historical blend results are not substituted for any extension control. Exact
build/checkpoint/config identity cannot be proved for the older runs, while the
rollout implementation is new. Each new rollout therefore has a fresh
same-build blend control. Historical blends are only secondary consistency
checks.

## Frozen configuration

All tasks use fixed-width MCTS, PUCT `0.1`, rollout horizon three, a 10,000
external-action cap, three workers, a six-hour per-instance limit, and the
same code commit as the existing subset.

Domain-appropriate historical search budgets are frozen:

| Domain | Width | Simulations | Teacher |
| --- | ---: | ---: | --- |
| Block Grouping | 5 | 20 | `hadd-gbfs` |
| Counters | 5 | 20 | `hmrmax-astar` |
| Drone | 20 | 70 | `hadd-astar` |
| FO Counters | 20 | 70 | `hmrmax-astar` |
| Rover | 20 | 70 | `hmrp-ha-gbfs` |
| MPrime | 20 | 70 | `hmrp-ha-gbfs` |

The MPrime rows use the corrected IPC-scale validation selection: training
jobs `20618726`/`20618727`, selected epochs 6/18. They intentionally do not use
the scientifically invalid old PRESERVE-4 validation endpoints.

## Interpretation

Two seeds are sufficient to report this direction as an exploratory dropped
configuration across every relevant imperfect domain. They are not enough for
a population-level superiority claim or admission to a primary RQ Holm family.
Report paired rollout-minus-blend coverage per domain and seed, runtime,
timeouts, action-limit failures, generated/peak nodes, evaluation time, and
rollout goal-hit rate. The learned-only and ENHSP-only arms remain a mechanistic
decomposition limited to Drone and FO Counters.

The all-domain screen has no performance promotion gate. Every declared
science task is released after compatibility succeeds, even if a smoke instance
is unsolved or rollout looks poor.

## Compatibility-only release

The extension uses one one-instance, 100-action rollout smoke per new domain.
These four tasks check only that the checkpoint loads, the domain invocation terminates,
the completion ledger is written, the expected evaluator is named, and a
successful plan (if any) passes VAL. The verifier never reads coverage or uses
outcomes to decide release.

Prepared commands (not executed by this preparation):

```bash
python scripts/verify_mcts_leaf_evaluator_all_imperfect.py \
  experiment_tracking/mcts_leaf_evaluator_all_imperfect_20260920/manifest.csv

# First-seed rollout rows for BG, Rover, Counters and MPrime.
sbatch --array=0,4,8,12 --cpus-per-task=2 --mem=20G --time=01:00:00 \
  --export=ALL,SMOKE=1 scripts/mcts_leaf_evaluator_all_imperfect_extension.sbatch

python scripts/verify_mcts_leaf_evaluator_all_imperfect.py \
  experiment_tracking/mcts_leaf_evaluator_all_imperfect_20260920/manifest.csv \
  --smoke-done /home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_all_imperfect/smoke/done \
  --write-compatibility /home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_all_imperfect/compatibility.json

# Release all 16 rows regardless of smoke coverage.
sbatch --array=0-15 scripts/mcts_leaf_evaluator_all_imperfect_extension.sbatch
```

For an unattended release, submit
`scripts/release_mcts_leaf_evaluator_all_imperfect_extension.sh` as a small
`afterok` job on the corrected four-smoke array. The release script is
idempotent, invokes the same compatibility-only verifier, and writes the
resulting science array ID to `science_submission.txt`.

## Resources and release checklist

The remaining science launch is exactly 16 tasks. Each requests 6 CPUs and
120 GiB for at most 72 hours, so a fully concurrent extension requests 96 CPUs
and 1,920 GiB (1.875 TiB). The four compatibility tasks request 8 CPUs and 80
GiB total for at most one hour. Across the existing subset and extension, the
complete 32-task design's maximum simultaneous request is 192 CPUs and 3,840
GiB (3.75 TiB), although scheduler concurrency may be deliberately capped.

Before release:

1. merge/deploy the manifest, verifier and runner into the isolated checkout;
2. verify the deployed checkout is exactly commit `2b243a6e...` or update all
   rows only after a reviewed same-build migration of the existing subset;
3. run the local verifier and focused tests;
4. confirm all eight new checkpoint directories exist and hash them;
5. run the four compatibility-only smokes and write the compatibility marker;
6. submit all 16 science identities, with no performance gate;
7. record the array job ID and immutable checkpoint hashes in a submission
   ledger; summarize only after all terminal/timeout outcomes are accounted for.

The runner uses the established broad MCTS exclusion
`ise-cpu-intl-[01,05,08-15,18,24-28]`. The narrower list was abandoned after
two compatibility tasks on node 25 and then two on node 12 exited with native
code `-4` before producing science. No scientific identity was duplicated as a
result; failed/cancelled smoke attempts remain provenance only.
