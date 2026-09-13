# MPrime Stage-1 Phase-B-A fixed-MCTS campaign

## Purpose

This is the missing Stage-1 fixed-search arm for MPrime. It evaluates the
checkpoint selected independently in each of the twenty Stage-1 lineages by
the frozen Phase-B replicate-A validator. It does not retrain a network and it
does not reuse the obsolete pre-Phase-B MPrime selections.

The local exact-identity audit found **0/20 reusable MCTS evaluations**. The
campaign therefore contains exactly twenty jobs: ten VH-off and ten VH-on.

## Frozen configuration

| Item | Value |
|---|---:|
| Checkpoint selector | Phase-B replicate A, per lineage |
| Test distribution | Original 20-instance MPrime test set |
| Retained children | 20 |
| Simulations per external action | 70 |
| PUCT coefficient | 0.1 |
| Leaf-estimator mixture | 0.5 |
| Terminal-safe action selection | Off; canonical historical-style fixed comparator |
| Workers per job | 3 |
| CPUs / RAM per job | 6 / 120 GiB |
| Per-instance timeout | 6 hours |
| External-action cap | 10,000 |
| Slurm walltime | 72 hours |

The manifests are separated by value-head mode so neither mode is a
dependency of the other:

- `manifest_off.csv`: array indices 0-9.
- `manifest_on.csv`: array indices 0-9.

Every row carries the exact checkpoint path, selected epoch, matching policy
score, source training/policy jobs and logs, and SHA-256 hashes of both source
selection ledgers.

## Recovery and validation

The runner uses rolling evaluation with a completion JSONL keyed by the stable
manifest identity. Successful and ordinary-unsolved instances therefore
survive a Slurm requeue or an exact resubmission. Every attempt has its own
immutable stdout log and VAL summary; `--allow-incomplete` preserves useful
validation evidence even if the evaluator exits before its final aggregate.
An append-only `attempts.tsv` joins the job/restart identity to its log,
completion record and VAL summary.

## Pre-submission sequence

1. Regenerate and diff the manifests:

   `python scripts/build_mprime_phase_b_a_stage1_mcts_20260913.py`

2. Run the regression tests:

   `python -m unittest asnets.tests.test_mprime_phase_b_a_stage1_mcts`

3. Deploy one exact commit to the isolated cluster checkout.
4. Run the one-instance compute smoke. It uses one worker, two CPUs, 40 GiB,
   instance 1 only, a ten-minute instance timeout and a 30-minute allocation.
5. Only after the smoke produces a completion record and VAL summary, submit
   the two independent ten-task arrays.

Example commands are intentionally not recorded with a mutable commit hash.
At deployment, set `CODE_COMMIT` to the exact deployed commit and submit the
smoke or full runner with `--export=ALL,CODE_COMMIT=<hash>,VALUE_HEAD=off|on`.

## Status

Compute smoke `21237283` passed on the isolated checkout. The full arrays are
running as `21237328` (VH-off, ten tasks) and `21237329` (VH-on, ten tasks).
At the 14 September 2026 00:21 IDT snapshot, all twenty tasks were live at
about 5h31m elapsed. No seed was terminal. Durable success lower bounds were
at least 7.2/20 VH-off and 7.7/20 VH-on; these are progress indicators, not
final means, and do not support a CI, p-value or RQ conclusion. With three
workers and twenty six-hour-capped instances, seven waves give an
evaluator-derived total bound of roughly 42 hours plus overhead; the
scheduler hard remainder was about 66h29m. Current result/log routes are indexed in
`../advisor_followup_20260910/live_result_progress_20260914_0021.csv`.
