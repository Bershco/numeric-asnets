# TPP first-update Phase D: matched legacy KL under the exact RNG stream

Status: implemented locally; compute smoke and four scientific tasks pending submission.

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
