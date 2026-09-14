# Block Grouping PW70 tie-mechanism trace

This is a logging-only diagnostic, not a coverage campaign or a tie-break
treatment test. It preserves the four original validation-selected Stage-1
checkpoints and exact PW70 configuration: terminal-safe search, `Kmin=3`,
`Kmax=20`, `c=0.6`, `alpha=0.5`, 70 simulations, PUCT `0.1`, estimator mixture
`0.5`, action-ID tie-breaking, 10,000 actions and six hours per instance.

The four predeclared targets are all among the 17 confirmed
policy-success/PW70-timeout cases. The sample contains exactly one target from
each original seed/VH cell and spans medium, large and extreme Block Grouping
instances. This is sufficient to ask whether PW first leaves a successful
policy path because of equal maximum visits, or whether unique visit maxima and
per-root search cost dominate. It is not representative enough to estimate a
domain score.

`--action-debug` records, at every external decision, elapsed time, the raw
network argmax/probability, selected action, root and edge visits, visit margin
and entropy, and every expanded child's `N`, `Q`, `U`, prior and raw-network
probability. The first divergence will be joined to the source policy plan
listed in `manifest.csv`.

The compute smoke uses target row 0 for at most three minutes of evaluator time.
The four scientific tasks are dependency-gated on that smoke and concurrency
limited to two. Each task uses one worker, 2 CPUs, 120 GiB and at most seven
hours. Therefore the experiment can occupy at most 4 CPUs and 240 GiB, with a
maximum two-wave allocation-runtime bound of 14 hours after the first two
tasks start, excluding any queue wait before or between waves.

Decision gate: expand or test a PW policy-prior treatment only if multiple
targets first diverge at an equal maximum-visit root and policy-prior would
select the successful pure-policy action. If divergences mostly have unique
visit maxima, or elapsed time is dominated by expensive roots before such a
tie, stop this branch.

## Submission

Static gates passed on the cluster: Python and shell syntax, all four exact
checkpoint directories, all four policy logs and all four historical PW70 logs
were present. Compute smoke `21265079` and dependent scientific array
`21265080[0-3]%2` were submitted on 14 September 2026. Exact resources,
dependency and output paths are recorded in `submissions.tsv`.

The first smoke reached the real evaluator but exited `-4` on
`ise-cpu-intl-11`; no scientific task started. The corrected scripts exclude
only that exact node and `ise-cpu-intl-13`, the other independently documented
native-exit node. They do not blacklist the wider node family. Failed or
superseded chains remain in `submissions.tsv`; all use the same frozen manifest
and scientific configuration.

Replacement smoke `21265430` then encountered the same pre-inference native
exit on `ise-cpu-intl-01`; dependent array `21265432` was cancelled before any
scientific task ran. The final scripts exclude the exact native-failure nodes
already supported by the MPrime controller (`intl-01`, `intl-09`, `intl-10`,
`intl-27`) plus Block Grouping's observed `intl-11` and known `intl-13`.
This remains a six-node compatibility exclusion, not a family blacklist.
Final smoke `21266623` and dependent scientific array `21266624[0-3]%2`
were submitted with that exact exclusion list.

Smoke `21266623` completed successfully in 5m11s. At the 19:05 IDT
snapshot, trace task 0 was running and tasks 1-3 were resource-pending. This
is normal capacity waiting, not a scientific hold. No mechanism conclusion is
drawn until the four traces are parsed against their exact source-policy plans.

At 14 September 23:19 IDT, tasks 0 and 1 were running at 4h34m and 3h59m;
tasks 2 and 3 were array-limit pending and will take their places. The active
footprint is 4 CPUs and 240 GiB. The first two outcomes are due no later than
approximately 02:20 IDT under their seven-hour allocations. If the second wave
starts immediately, all four allocations end by approximately 09:20 IDT;
queue handoff delay can move that bound later. No terminal trace or mechanism
conclusion was available at the snapshot.
