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
then Holm is recomputed within the intended final six-domain family.  The
stored RQ1/RQ3 evidence currently keeps MPrime in a separate extension file;
those files must be merged and the real six-domain Holm values recomputed
before the final thesis tables.  RQ2/RQ4 already use six-domain fixed-search
families.

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
