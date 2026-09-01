# MPrime anchor-validation stability audit

The corrected rescore is complete at 588/588 points and selects anchor 10 for
both VH modes. This fixes the invalid earlier anchor-zero result, but it does
not prove that a single frozen validation sample ranks anchors robustly.

The VH-off AUC margin between anchor 10 and 30 is 0.003969. The VH-on margin
between anchor 10 and anchors 3/30 is only 0.000794. The latter is sufficiently
small that validation-sample stability must be reported.

Recommended diagnostics, none of which retrains the 28 tuning lineages:

1. Bootstrap the frozen validation instances and report how often each anchor
   wins, with confidence intervals for pairwise AUC differences.
2. Generate one independent, versioned IPC-scale validation replicate before
   inspecting its scores; rescore the same 588 saved checkpoints and compare
   winner/rank stability.
3. Stratify both sets by structural difficulty and repeat selection under
   leave-one-band-out analysis.
4. Report validation-versus-test rank agreement only as a diagnostic; never use
   test coverage to choose the coefficient.

The preferred test is bootstrap plus one independent frozen replicate. Stage-2
jobs remain valid while this measures coefficient-selection robustness.
