# Counters strict-confirmation divergence audit

Snapshot: 14 September 2026, approximately 18:08 IDT. The two strict jobs
were still live, so this is an exact audit of classified records at that
snapshot rather than a terminal campaign result.

## Scope and denominators

This audit addresses two validation-selected Stage-1 VH-off checkpoints:

| Seed | Epoch | Pure-policy coverage | Action-ID classified | Policy-prior classified |
|---:|---:|---:|---:|---:|
| 534933607 | 43 | 59/59 | 31/59 | 35/59 |
| 2082152039 | 12 | 35/59 | 36/59 | 42/59 |

The first checkpoint's policy solves every instance. The second solves only
35/59, so only those 35 instances can be policy-success/search-failure
opportunities. Search failures on the other 24 instances are deliberately
excluded from this audit.

`per_instance_divergence.csv` contains one row for every currently classified
policy-success/search-failure arm. `per_instance_pair_join.csv` contains every
pure-policy success, including successful or still-unclassified search arms.
Both carry the original remote log paths. `raw_file_checksums.csv` makes the
locally cached source evidence reproducible.

## Exact findings

Every one of the 58 currently classified policy-success/search-failure arm
outcomes was an ordinary 10,000-action termination. None was a six-hour
per-instance timeout, OOM, invalid plan, or scheduler-truncated classification.

| Seed | Rule | Failures audited | Divergence at decision 1 | At decision 2 | Later | Terminal cause |
|---:|---|---:|---:|---:|---:|---|
| 534933607 | Action ID | 11 | 5 | 6 | 0 | 11 ordinary 10k caps |
| 534933607 | Policy prior | 16 | 6 | 10 | 0 | 16 ordinary 10k caps |
| 2082152039 | Action ID | 15 | 14 | 0 | 1 | 15 ordinary 10k caps |
| 2082152039 | Policy prior | 16 | 0 | 5 | 11 | 16 ordinary 10k caps |

### Seed 534933607

Among the 31 instances classified by both rules, 19 succeed under both, 11
fail under both, and one (`fz_instance_22.pddl`) succeeds under Action ID but
fails under policy-prior tie-breaking. All eleven shared failures choose the
same non-policy action at the same first or second decision under both rules.
This is consistent with the pure-policy action not belonging to the
maximum-visit set, in which case the tie-break has no opportunity to repair
the decision. It is not proven by these logs: the strict jobs did not emit the
full root vectors, and separate processes can have small numerical
differences. The evidence therefore cannot distinguish a unique non-policy
winner, a tie among other non-policy actions, or apportion the override
between Q and the PUCT exploration term.

The one directly observed paired regression is `fz_instance_22.pddl`:
Action-ID search succeeds, while policy-prior search diverges at decision 2
and reaches the 10,000-action cap. It must remain visible in the final matched
coverage result rather than being dismissed as noise.

The remaining 28 policy-success instances have at least one unclassified arm
and are not compared prematurely.

### Seed 2082152039

Across its 35 pure-policy successes, 19 currently succeed under both search
rules. Fourteen of the fifteen overlapping failures exhibit a clear
intervention pattern: Action ID first diverges at decision 1, whereas
policy-prior tie-breaking postpones those same eventual failures to a median
decision of 70 and as late as decision 259. Since tie-breaking is the intended
treatment difference, this is strong evidence that policy-prior resolution
preserved the pure-policy path at the early tie. Full root vectors were not
logged, so the audit reports the behavior as a delayed divergence rather than
claiming a directly observed visit vector.

The rescue is nevertheless temporary. At the policy-prior arm's later first
divergence, policy-prior tie-breaking no longer preserves the pure-policy
action. A unique visit maximum or another action-selection override is likely,
but the missing root vector prevents a finer causal label. The trajectory then
leaves the successful policy path and eventually reaches 10,000 actions. One additional shared failure
(`fz_instance_24.pddl`) diverges at the same decision under both rules, so the
tie rule has no effect there. One policy success remains unclassified under
Action ID but is a classified failure under policy-prior search.

## Defensible conclusion

The low coverage is not explained by timeouts or OOM. It is caused by early
departures from policy followed by long, non-goal 10,000-action trajectories.

The two checkpoints fail for meaningfully different proximate reasons:

1. For the perfect-policy seed 534933607, both rules usually make the same
   non-policy choice within two decisions. That is consistent with the policy
   action losing the visit-count contest before tie-breaking, but a root trace
   is needed to prove the exact visit topology.
2. For seed 2082152039, policy-prior tie-breaking often fixes the initial
   arbitrary tie and preserves the policy trajectory substantially longer,
   but later divergence still leads to the 10,000-action cap. Because these
   strict-run logs do not contain full root visit vectors, they cannot
   determine whether those later divergences came from a unique visit winner,
   a maximum-visit tie, or another action-selection path.

Accordingly, the targeted 3/3 Counters rescue remains a real mechanistic
result, but it is not sufficient as a domain-wide repair. The strict campaign
must finish before estimating a matched coverage effect. A deeper causal test
of the remaining failures would need selected per-root traces that record the
complete visit/Q/U/prior vector and the goal-chase/eligibility path at the
first policy-prior divergence. The present logs establish when and how the
external trajectories diverge, but not the exact internal override.

## Files

- `per_instance_divergence.csv`: failing-arm records, first divergence, two
  separate cause fields, and original log provenance.
- `aggregate_causes.csv`: seed/rule × first-divergence mechanism × terminal
  cause counts.
- `per_instance_pair_join.csv`: paired state for every pure-policy success.
- `aggregate_pair_mechanisms.csv`: paired outcome/mechanism counts.
- `raw_file_checksums.csv`: local evidence hashes.
- `analyze.py`: deterministic reconstruction script.
