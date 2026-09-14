# MPrime Stage-1 Phase-B-A PW70 screen

## Scientific question

Can progressive widening retain the useful coverage of MPrime's canonical
Stage-1 fixed 20/70 MCTS while opening substantially fewer children early in
search?

This is a four-task screen, not a confirmatory ten-seed result. The two seeds
`1963100312` and `2011206605` were predeclared because they are the same matched
two-seed screen identities used in the earlier cross-domain PW work. Both
VH-off and VH-on are evaluated, yielding exactly four domain/mode/seed cells.

## Frozen identity

Each row is derived directly from the corresponding canonical fixed-MCTS row
in `../mprime_phase_b_a_stage1_mcts_20260913/manifest_{off,on}.csv`. The
checkpoint, Phase-B-A selector, policy provenance, 70 simulations, PUCT 0.1,
estimator mixture 0.5, six-hour per-instance timeout, 10,000-action cap,
workers, CPUs, memory and 72-hour allocation are unchanged.

The only algorithmic change is progressive widening with:

- `Kmin = 3`;
- `c = 0.6`;
- `alpha = 0.5`.

Terminal-safe action selection remains disabled so the comparison matches the
canonical fixed-MCTS arm rather than silently combining PW with a second
safety intervention.

## Resources and outputs

- four array tasks;
- 6 CPUs and 120 GiB per task;
- maximum concurrent request: 24 CPUs and 480 GiB;
- three workers per task;
- six hours per instance and 72 hours per task;
- durable per-instance completion JSONL, immutable attempt logs, append-only
  attempt ledger and per-attempt VAL summary.

The full manifest is `manifest.csv`. Submission provenance is recorded in
`submissions.tsv`; the compute smoke is recorded in `smoke.tsv`.

## Deployment and status

The reviewed implementation was pushed as commit
`9fe4f9e695ed290cc370dd5c5b877c276578525c` and deployed to a new detached
cluster worktree at
`/home/hersco/bershco-nu-asnets/numeric-asnets-mprime-pw-9fe4f9e6`. This did
not modify the dirty isolated checkout used by existing work. The production
native TensorFlow operator is linked into the detached checkout.

All four local and compute-node controller tests pass. Compute-smoke history is
kept rather than hidden:

- `21240242` ran on the already known incompatible node
  `ise-cpu-intl-13` and exited `-4` before inference. It produced no scientific
  result.
- `21240249` carried a malformed commit export, failed in three seconds before
  inference and is discarded as an orchestration error.
- corrected smoke `21240250` excluded only `ise-cpu-intl-13`, solved the real
  MPrime instance in 28.44 seconds, produced a durable completion record and
  passed VAL (`1/1` valid, zero invalid).

The scientific array is `21240256[0-3]`, submitted at 00:15 IDT on 14 September
2026. All four tasks entered `RUNNING` immediately. Each requests 6 CPUs,
120 GiB and 72 hours; the maximum concurrent request is 24 CPUs and 480 GiB.
Only `ise-cpu-intl-13` is excluded. Exact row/job/log mappings are in
`submissions.tsv`, and smoke provenance is in `smoke.tsv`.

## Completed result

All four tasks completed. VH-off scores are 20/20 and 17/20 versus matched
policy scores 20/20 and 13/20: mean policy 16.5, mean PW70 18.5, mean change
+2.0. VH-on scores are 18/20 and 20/20 versus 17/20 and 16/20: mean policy
16.5, mean PW70 19.0, mean change +2.5.

This is a promising two-seed screen, not confirmatory evidence. With two
matched seeds no useful confidence interval or significance claim is made,
and the still-incomplete fixed-MCTS recovery prevents a clean PW-versus-fixed
comparison. Exact results, elapsed times and source paths are in
`results_20260914.csv`.

## Ten-seed confirmation extension

The promising two-seed result is being expanded to the complete matched
ten-seed Stage-1 Phase-B-A cohort. The extension contains exactly the other
eight canonical seeds in each VH mode (16 tasks); it excludes the two completed
screen seeds and therefore performs no duplicate scientific evaluation.

`manifest_confirmation_remaining.csv` is the authoritative 16-row manifest.
The unchanged compute-smoked runner requires four rows per array, so runtime
copies are split across `manifest_confirmation_part1.csv` through
`manifest_confirmation_part4.csv`. These copies preserve each authoritative
row byte-for-byte except for the local array index. All rows inherit the exact
checkpoint, selector hash, fixed-comparator manifest hash, policy provenance,
and resource/search identity already frozen for the canonical fixed-MCTS arm.

The four unthrottled, low-priority arrays were submitted at 15:10 IDT on 14
September 2026:

- `21259752[0-3]`: VH-off, seeds 534933607, 923500475, 1073581256,
  1239739722;
- `21259753[0-3]`: VH-off, seeds 1472491096, 1510771779, 1972442430,
  2082152039;
- `21259754[0-3]`: VH-on, seeds 534933607, 923500475, 1073581256,
  1239739722;
- `21259755[0-3]`: VH-on, seeds 1472491096, 1510771779, 1972442430,
  2082152039.

Each task requests 6 CPUs, 120 GiB, and at most 72 hours. If all sixteen run
simultaneously, the extension requests 96 CPUs and 1,920 GiB. Immediately
before submission the verified live footprint was 180 CPUs and 4,140 GiB, so
full concurrent admission would reach 276 CPUs and 6,060 GiB—84 GiB below the
observed 6 TiB account ceiling. Slurm initially left all sixteen pending for
cluster resources. `Nice=10000` gives this extension lower priority than the
already-running work; there is no artificial array concurrency throttle. The
known incompatible node `ise-cpu-intl-13` remains the only excluded node.

Exact row-to-job, log, output, commit and runtime-manifest provenance is in
`confirmation_submissions.tsv`. The hard completion bound is 72 hours after
each task actually starts; no defensible queue-start estimate is available.

Task `21259754_3` (VH-on seed `1239739722`, selected epoch `20`) failed
before inference with native evaluator exit `-4` on `ise-cpu-intl-11`; it
produced zero classifications. Exact replacement `21264782_3` uses the same
manifest row and immutable checkout while excluding only
`ise-cpu-intl-11` and the already documented incompatible
`ise-cpu-intl-13`. No completed or running identity was duplicated.

The final analysis must report, for each VH mode, policy coverage, fixed-search
coverage at 30 minutes / 2 hours / 6 hours, and PW70 coverage at the same three
cutoffs. It must include paired PW-minus-policy and PW-minus-fixed confidence
intervals and exact raw/Holm-adjusted p-values; a PW improvement is not claimed
from policy comparison alone.

At 14 September 23:21 IDT, fourteen of the sixteen confirmation-extension
tasks had left the queue and two were still running: VH-off seed `1073581256`
and replacement VH-on seed `1239739722`. Together with the four completed
screen identities, every ten-seed cell has durable evidence. Conservative
means, counting every not-yet-classified instance as failure, are:

| VH | Policy | Fixed 30m / 2h / 6h | PW70 lower bound 30m / 2h / 6h |
|---|---:|---:|---:|
| off | 16.3 | 13.0 / 14.7 / 15.7 | >=15.7 / >=17.6 / >=17.7 |
| on | 15.7 | 13.3 / 15.1 / 16.0 | >=14.5 / >=16.8 / >=17.4 |

PW70 therefore already exceeds fixed search at every cutoff and exceeds policy
by two hours in both modes. It does not yet exceed policy at 30 minutes. The
remaining durable ledgers contain four and thirteen unclassified instances;
with three workers and a six-hour per-instance cap, their evidence-based
remaining workloads are at most roughly 12 and 30 allocation-hours,
respectively. Final confidence intervals and exact tests remain gated on full
terminal reconciliation.
