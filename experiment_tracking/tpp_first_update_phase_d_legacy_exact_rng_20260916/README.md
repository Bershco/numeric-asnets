# TPP first-update Phase D: matched legacy KL under the exact RNG stream

Status: one-epoch treatments complete; endpoint evaluations submitted. Compute smoke `21408030` completed.
The first training array `21408031_[0-1]` failed before scientific execution
because the detached checkout lacked the ignored compiled TensorFlow operator;
its dependent endpoints `21408032_[0-1]` cancelled without work. The production
operator was then linked into the isolated checkout and its SHA-256
`a9ab808dbe131469f25d952864e41b4d53e10aa2b0948eb0f6667b5af80b71c6`
verified. Exact replacement training `21408668_[0-1]` completed both scientific
epochs and materialized both checkpoints, then its post-run verifier failed on
an invalid requirement that task gradients remain identical after the first
treatment-dependent update. The corrected verifier checks the same sixty
frozen batches and RNG seeds throughout, and exact equality of the task-policy
gradient and argmax pattern at the pre-treatment step. Both outputs pass. No
retraining is needed. Endpoint array `21409234_[0-1]` now evaluates the two
verified checkpoints. No scientific result from the failed first array is used.

## Why this experiment exists

The historical catastrophic seed genuinely scored 9/20 at its selected Stage-2 checkpoint, and the independently reproduced first Stage-2 checkpoint scored 10/20. The Phase-C exact-RNG inactive arm genuinely scored 20/20, but it was not the historical treatment: it changed both the stochastic stream and the current-policy forward used by the anchor KL.

The Phase-C inactive arm therefore cannot establish that RNG alone prevented the collapse. At the first update, the historical/dropout-current anchor gradient had norm 3.32865 and cosine +0.831 with the replay-policy gradient; deterministic-current KL reduced the anchor gradient to approximately 1.57e-7.

## Minimal matched comparison

This experiment reuses Phase C's deterministic-current, RNG-314159 inactive results and adds only:

- catastrophic seed 1972442430, legacy/dropout-current KL, fixed optimizer-step RNG 314159;
- stable seed 2082152039, the same treatment;
- one policy endpoint evaluation for each.

Both jobs reuse the exact 60 frozen replay batches, source checkpoints, learning rate 0.0003, constant anchor coefficient 3, and task/dropout RNG stream from Phase C. The implementation records and verifies the fixed per-step RNG without enabling rollback. At every optimizer step it requires the task-policy gradient norm and complete target-versus-policy argmax-disagreement pattern to match Phase C. The older Phase-C audit did not serialize full policy tensors, so this is a direct but not bitwise-complete forward-equivalence check.

## Decision rule

- If the matched legacy-KL catastrophic arm collapses while Phase C deterministic KL stays 20/20, dropout-contaminated anchor KL is the demonstrated rescue mechanism under this RNG stream.
- If matched legacy KL also stays near 20/20, RNG-stream susceptibility is sufficient; proceed to a small, predeclared multi-RNG screen.
- The stable control checks that the treatment does not manufacture a failure in an already stable lineage.

No 100-epoch training is part of this diagnostic. The broader TPP/Drone guard campaign remains held until a matched inactive condition reproduces or explains the failure.
