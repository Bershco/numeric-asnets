# Coverage-ceiling significance audit — 19 September 2026

This audit asks a deliberately extreme question for every primary direct-effect
cell: if the comparison method solved every test instance in every seed, could
the existing paired sample pass the two-sided exact sign-flip test and its
declared Holm family?

The result is a resolution diagnostic, not an achieved result, a power
calculation, or permission to select methods using the test set.

For a perfect comparison score, every baseline seed below the domain ceiling
has a positive paired difference and every already-perfect baseline seed is a
zero tie.  If `k` seed pairs remain nonzero, the best attainable two-sided
exact sign-flip p-value is

`p_min = 2^(1-k)`.

Thus a high mean baseline is not itself the statistical obstruction.  The
obstruction is seed-level ceiling ties.  Small remaining headroom still limits
the practical effect size and makes ties more likely, but it does not increase
the exact p-value when all nonzero differences share a sign.

The CSV contains target-wise counterfactuals: one domain is replaced by
perfect coverage while every other hypothesis keeps its observed raw p-value,
then Holm is recomputed within the final six-domain family. RQ1/RQ3 and
RQ2/RQ4 now all use the merged MPrime-inclusive six-domain families.

## Main conclusion

- With six hypotheses, eight nonzero same-direction pairs guarantee a
  best-case Holm value no worse than `0.046875`.
- Block Grouping, Drone, FO Counters and Rover retain ten headroom pairs in the
  direct cells audited here, so perfect coverage would pass comfortably.
- MPrime retains eight to ten effective pairs depending on the cell, so its
  relatively high coverage does not make significance impossible.
- Counters Stage-1/VH-off retains only seven headroom pairs.  In RQ1, even
  perfect Stage-2 coverage gives raw `p=0.015625` and six-domain Holm
  `p=0.09375`; the present ten nominal seeds cannot pass that family.
- The same Counters baseline gives a best-case RQ2 Stage-1 Holm value of
  `0.046875` at 30 minutes but `0.078125` at two and six hours because its Holm
  rank changes relative to the other observed hypotheses.
- One additional *non-ceiling* paired Counters seed would raise `k` from seven
  to eight.  An added seed already at the baseline ceiling would not improve
  the exact-test resolution.

See `perfect_coverage_counterfactual.csv` for every audited RQ/stage/cutoff.

## Favorable baseline-plus-one resolution counterfactual

Perfect coverage is an extreme resolution bound, not a sensible performance
target for difficult domains such as Rover. The companion file
`minimum_consistent_gain_for_holm.csv` asks a narrower resolution question:
what is the smallest favorable, perfectly consistent *baseline-plus-one*
pattern that would pass the six-domain Holm family?

For each row, the audit constructs a hypothetical comparison that equals the
baseline plus one solved instance in `k` distinct non-ceiling seeds and equals
the baseline in every other seed. It does **not** add those solves to the
observed comparison result. The audit recomputes the exact paired sign-flip
p-value, replaces only that raw p-value in the observed six-domain family and
recomputes Holm. The old label “minimum target mean” was misleading and has
been replaced by `counterfactual_mean_under_baseline_plus_one_pattern`.

| Cell | Baseline | Passing baseline-plus-one counterfactual mean | Positive pairs |
|---|---:|---:|---:|
| RQ1 Rover, Stage 2 policy | 4.0/20 | 4.8/20 | 8/10 |
| RQ3 Rover, VH-on refinement | 3.8/20 | 4.6/20 | 8/10 |
| RQ2 Rover, Stage-1 MCTS, 30m | 4.0/20 | 4.7/20 | 7/10 |
| RQ2 Rover, Stage-1 MCTS, 2h/6h | 4.0/20 | 4.8/20 | 8/10 |
| RQ1 MPrime | 16.3/20 | 17.1/20 | 8/10 |
| RQ3 MPrime | 15.7/20 | 16.5/20 | 8/10 |
| RQ3 Counters | 18.6/59 | 19.4/59 | 8/10 |

These values are not coverage targets for the observed methods. Rover
illustrates why: its observed Stage-1 MCTS mean already exceeds some listed
counterfactual means, yet inconsistent seed-wise directions keep the actual
exact test nonsignificant. Exact paired inference depends on where gains and
losses occur, not only on the mean.

There is consequently no unique answer to “what mean must the current method
reach?” without freezing an allocation rule for additional solves. Any future
current-result continuation audit must separately declare whether solves are
allocated to the most negative pairs, to distinct zero pairs, uniformly, or
by another predeclared rule. The perfect-coverage bound and this favorable
baseline-plus-one pattern remain useful as resolution diagnostics only.

Holm is a multiple-testing correction, not the underlying paired test. The
canonical report keeps paired effect sizes and confidence intervals primary,
reports raw exact p-values, and treats Holm as familywise-error control.
Legitimate sensitivity analyses include dependence-aware maxT or
Westfall--Young correction, or a predeclared hierarchical paired bounded-count
model. Benjamini--Hochberg is appropriate only for an explicitly exploratory
false-discovery family. None should be selected post hoc merely because it
makes a result significant, and per-instance tests remain invalid because the
same instances are shared within seeds.
