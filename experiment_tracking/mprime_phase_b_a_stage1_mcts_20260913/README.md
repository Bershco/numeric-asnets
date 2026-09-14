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

## Final status — 14 September 2026

Compute smoke `21237283` passed. Full arrays `21237328` and `21237329`
evaluated the twenty final-validator checkpoints. Fifteen instances were left
without terminal records when their original allocations ended; the exact
recovery campaign ran only those identities. All fifteen reached their
six-hour per-instance timeout. Three initial recovery tasks failed before
inference and were replaced exactly; no scientific identity was duplicated.

The locally reconciled result is complete: `400/400` instances classified,
`317` unique successes, `82` six-hour timeouts and one ordinary-unsolved
result. All `317` unique successful plans are covered by VAL-valid records
(the raw VAL files contain one duplicate validation row, hence 318 rows).

| VH | Policy | Fixed MCTS 30m / 2h / 6h | Six-hour change [95% CI] | Raw / provisional six-domain Holm p |
|---|---:|---:|---:|---:|
| Off | 16.3/20 | 13.0 / 14.7 / 15.7 | -0.6 [-2.00, 0.80] | .445 / .500 |
| On | 15.7/20 | 13.3 / 15.1 / 16.0 | +0.3 [-1.70, 2.30] | .828 / .984 |

At 30 minutes, VH-off fixed search is significantly below its policy baseline
after provisional six-domain Holm correction: `-3.3 [-5.18,-1.42]`, raw
`p=.0098`, Holm `p=.0391`. By six hours neither mode differs reliably from its
own policy. The VH interaction is not reliable at any cutoff.

Auditable outputs are in `results_20260914/`: `per_instance_results.csv`
contains the exact evaluation job and original cluster log/completion route;
`per_seed_results.csv` joins training, policy and checkpoint provenance; and
`mprime_rq_extension.csv` records the paired RQ2/RQ4 estimates.
