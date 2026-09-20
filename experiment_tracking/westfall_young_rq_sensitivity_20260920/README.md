# Dependence-aware maxT sensitivity analysis — 20 September 2026

This directory contains an explicitly specified **post-hoc** sensitivity
analysis for the primary validation-led RQ families. It was added after the
canonical results were known, so it does not replace the paired exact tests
with Holm familywise correction.

The six domains share the same ten seed identifiers. The analysis enumerates
all `2^10 = 1024` synchronized sign flips: one sign is sampled per seed and
applied to every domain simultaneously. This preserves the observed
cross-domain dependence. The statistic is the absolute studentized paired
mean difference. Both single-step Westfall--Young/maxT and step-down maxT are
reported for every six-domain estimand/cutoff family.

The implementation is `scripts/build_westfall_young_rq_sensitivity_20260920.py`.
The complete 162-row output is `westfall_young_maxT_results.csv`.

## Interpretation

The dependence-aware correction changes several borderline conclusions at
alpha `.05`:

| Family/cell | Holm | maxT single-step | maxT step-down |
|---|---:|---:|---:|
| RQ2 Stage 2, 30m, MPrime VH-off direct (harm, -2.3) | .0625 | **.04297** | **.02930** |
| RQ3 Block Grouping VH-on direct (harm, -3.1) | .09375 | **.02734** | **.02734** |
| RQ3 Block Grouping VH interaction (harm, -2.8) | .10547 | **.04492** | **.04492** |
| RQ4 Stage 1, 30m, MPrime VH-on direct (harm, -2.4) | .07031 | .05078 | **.02734** |
| RQ4 Stage 2, 30m, Block Grouping interaction (benefit, +1.9) | .07813 | **.04297** | **.03516** |

RQ1 FO Counters remains nonsignificant: Holm `.28125`, single-step maxT
`.11914`. Therefore maxT is not being selected merely because it produces
smaller p-values; it is less conservative for some strongly correlated
families and more conservative for others.

Report effect sizes and confidence intervals first, retain Holm as the
canonical analysis, and label these results as a dependence-aware robustness
analysis. The family definitions, synchronized seed mapping, and use of
studentized statistics are frozen in this artifact and must not be changed in
response to its significance pattern.
