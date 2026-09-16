# Historical Stage-2 anchor-gradient availability audit

Scope: the exact 100 validation-led Stage-2 lineages used by the primary five
imperfect domains: Block Grouping, Drone, FO Counters, Rover and Counters,
crossed with VH-off/VH-on and ten seeds.

A literal-path remote audit confirmed that all 100 training logs exist. A
marker-only scan found zero occurrences of the three fields needed for a
decomposed gradient analysis: `policy_gradient_l2`,
`weighted_anchor_gradient_l2`, and `policy_anchor_gradient_cosine`.

The result is 0/100 lineages with retrospectively measurable decomposed anchor
gradients. It is **not** evidence that zero lineages were affected. The fields
were introduced only in the 15 September first-update instrumentation and are
emitted only with `ASN_FIRST_UPDATE_AUDIT_PATH`; the historical campaigns
predate it.

Separately, all twenty final MPrime Stage-2 training logs exist and the same
marker-only audit found 0/20 with decomposed gradient fields. MPrime is not
silently folded into the primary 100-lineage denominator.

Raw anchor KL, total loss, and total/clipped gradient are deliberately not used
as proxies. They cannot isolate whether the KL branch itself produced a large
or misdirected gradient. Therefore the old logs cannot rank cross-domain
severity or justify broad Stage-2 retraining. A future decision requires a
small instrumented first-update screen on frozen Stage-1 checkpoints before
any full retraining.

Row-level job/log routes are frozen in
`stage2_anchor_gradient_availability_20260916.csv`; the aggregate is
`stage2_anchor_gradient_availability_20260916.json`.
