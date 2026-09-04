# Node-local semaphore ENOSPC / native SIGILL incident — 2026-09-04

## Scope

Two infrastructure-only failure modes occurred before inference. On `ise-cpu128-03/04`, multiprocessing queue creation raised `OSError: [Errno 28] No space left on device` from `_multiprocessing.SemLock`. On a distinct set of 40-core `ise-cpu-intl` nodes, the native evaluator process exited `-4`, i.e. POSIX signal 4 (`SIGILL`). No model evaluation, plan, or scientific zero score was produced by either failure mode.

## Proven nodes

- `ise-cpu128-03`: repeated TPP policy and initial branch-completion failures.
- `ise-cpu128-04`: repeated TPP policy, Block Grouping, FO Counters, and Rover failures.
- `ise-cpu-intl-08,09,10,13,14,26,28`: directly observed native/evaluator `SIGILL` failures. These nodes advertise 40 CPU cores. Successful same-day jobs ran on 48–72-core `ise-cpu-intl` nodes, so only the evidenced nodes—not the whole family—are excluded.

No whole node family is blacklisted. Other `ise-cpu128-*` and `ise-cpu-intl-*` nodes continue to run the same workloads successfully.

## Affected scientific work and response

- Delivery: twelve previously scoreless curve/endpoint identities were rerun exactly and all twelve scored.
- TPP/off: 39 original scoreless rows identified; both selected endpoints are recovered. At the latest reconciliation, 192/213 unique curve identities score and 21 exact retries run.
- TPP/on: 43 original scoreless rows identified and exactly retried. At the latest reconciliation, 204/214 unique curve identities score; seven run and three exact third retries are submitted. All ten selected endpoints are recovered.
- Block Grouping/FO Stage-2 MCTS completion: nine initial failures exactly recreated; four second retries created where `ise-cpu128-04` caused another pre-inference failure.
- Rover Stage-2 MCTS completion: thirteen of nineteen initial jobs failed on `ise-cpu128-04`; seven first retries then failed with `SIGILL` on the exact `intl` nodes above. Every one of the nineteen intended identities now has a running or pending original/replacement job.

## Controller repair

Setting `SBATCH_EXCLUDE` in the environment of the outer submit wrapper was not reliable for the wrapper's nested `sbatch`. Both submission controllers now:

1. preserve `excluded_nodes` in the manifest;
2. submit the exact job;
3. run `scontrol update JobId=... ExcNodeList=...`;
4. verify the resulting `ExcNodeList` before recording the submission as successful.

This is operational filtering only. It does not change model, checkpoint, seed, search width, simulation count, instance budget, or evaluation set.
