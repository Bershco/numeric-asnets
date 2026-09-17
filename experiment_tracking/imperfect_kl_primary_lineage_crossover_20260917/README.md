# Imperfect-domain primary-lineage KL crossover

## Question

Across the actual ten validation-selected Stage-1 VH-off networks in each of
the five imperfect domains, how often does the historical dropout-current KL
change the first Stage-2 epoch relative to deterministic-current KL?

The earlier screen was intentionally exploratory and availability-driven. It
contained ten optimizer/replay schedules but only six source networks: Block
Grouping, Drone, FO Counters and Rover each reused one older tuning checkpoint
twice, while Counters used two primary checkpoints. It proved the gradient
mechanism but cannot estimate prevalence across independent primary networks.

## Frozen design

- 5 domains x 10 actual MAIN-VAL Stage-1 VH-off validation-selected networks.
- Exact starting-policy coverage is already frozen for every source checkpoint.
- 60 optimizer updates, coefficient 3, learning rate 0.0003, no guard or
  adaptive controller.
- Within every new pair, the legacy arm captures the exact 60 replay batches
  and targets that it consumes. The deterministic-current arm starts from the
  identical checkpoint and replays those files with the identical fixed
  optimizer RNG. Thus only KL current-policy semantics differ.
- A fail-closed verifier requires source-checkpoint hashes, ordered replay
  hashes, per-step RNG, step-0 policy-gradient norm and target-disagreement bits
  to match before endpoint evaluation.
- The two exact primary Counters pairs from the completed frozen crossover are
  reused after checkpoint-hash verification. The other 48 pairs are new.

The legacy capture is also the legacy treatment. This is not an asymmetric
shortcut: both paths invoke the same replay-update implementation, and the
fixed per-step RNG makes target generation unable to perturb optimizer/dropout
randomness. Existing capture-versus-frozen-legacy records are checked for
harness equivalence before the new chain is admitted. If that check fails, the
chain fails closed and symmetric legacy replay tasks must replace the captures.

## Submitted scope and gate

The immediate sequence is 48 legacy capture/treatment tasks, 48 paired
deterministic-current replay tasks, one verifier, and 96 policy endpoints.
There is no artificial Slurm concurrency throttle; the scheduler enforces the
cluster's running-resource envelope and may leave excess work pending. Known
bad nodes are excluded.

This estimates one-epoch susceptibility only. It does **not** authorize the
gated 100-epoch comparison or downstream MCTS. After completion, choose one
stable/cheap domain and one high-effect/high-variance domain for a ten-lineage
100-epoch comparison. Fixed and PW MCTS are needed only if corrected-KL policy
evidence makes the corrected method a candidate primary pipeline.

`primary_lineages.csv` contains all 50 source identities and starting scores.
`new_pairs.csv` contains the 48 newly executed pairs. Remote submission IDs and
verified checkpoint hashes are recorded alongside the run, then mirrored here.

## Submission

The preflight verified all 50 source checkpoint hashes and matched both reused
Counters hashes to the earlier frozen manifest. It also established harness
equivalence: for all ten earlier schedules, capture-legacy and frozen-legacy
used identical ordered batches/RNG and produced byte-identical exported model
weights after every one of the 60 updates. Eight full checkpoint files were
also byte-identical; the other two differed only in serialized TensorFlow
object identifiers.

| Role | Job | Tasks | Per-task request | Dependency |
|---|---:|---:|---:|---|
| Exact old-source baselines | 21434691 | 4 | 5 CPU / 20 GiB / 2h | none |
| Retry for operationally failed baseline tasks 1–2 | 21434754 | 2 | 5 CPU / 20 GiB / 2h | none |
| Legacy capture/treatment | 21434692 | 48 | 6 CPU / 120 GiB / 6h | none |
| Deterministic-current replay | 21434693 | 48 | 6 CPU / 120 GiB / 4h | matching capture |
| Pair verifier | 21434694 | 1 | 2 CPU / 8 GiB / 30m | both treatment arrays |
| Policy endpoints | 21434695 | 96 | 5 CPU / 20 GiB / 2h | verifier |
| Statistical finalizer | 21434696 | 1 | 1 CPU / 4 GiB / 30m | all endpoints |

The finalizer writes source → legacy → deterministic rows, paired 95% CIs,
exact sign-flip p-values and Holm corrections. Nodes `intl-14` and `intl-15`
were added to the exclusion list after both produced native `-4` failures in
the baseline array; the exact failed identities were resubmitted, not the
whole array.
