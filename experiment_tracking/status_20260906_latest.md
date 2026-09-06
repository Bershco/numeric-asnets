# Complete experiment snapshot — 2026-09-06T12:31:30+03:00


## Current workload
| state | jobs | cpus | memory_gib |
| --- | --- | --- | --- |
| RUNNING | 43 | 254 | 5048.0 |
| PENDING | 1 | 1 | 1.0 |


| experiment | running | pending | requested_cpus | requested_gib |
| --- | --- | --- | --- | --- |
| Counters exact-snapshot PW70 divergence recovery | 2 | 0 | 12 | 240.0 |
| FO Counters terminal-led Stage-2 MCTS | 2 | 0 | 12 | 240.0 |
| MPrime Phase B validation | 1 | 1 | 3 | 9.0 |
| PW70 ten-seed FO/Rover expansion | 20 | 0 | 120 | 2400.0 |
| PW70 two-seed correction | 1 | 0 | 6 | 120.0 |
| Stage-2 MCTS branch completion — Counters terminal-led | 13 | 0 | 78 | 1560.0 |
| Stage-2 MCTS branch completion — FO Counters validation-led | 4 | 0 | 24 | 480.0 |


## Actions and integrity findings
20 approved PW70 jobs submitted21039201–21039220; each6CPUs120GiB72h. MPrime planner array21039224 has six2CPU8GiB90minute tasks. Finalizer21039342 is afterok-dependent and will freeze sets, run a two-replicate checkpoint preflight, then release60 validation-only lineage jobs with at most12 concurrent.

Three partial jobs20974363,20863187,20838267 passed post-hoc VAL in job21039344:22,21,29 unique valid successes, zero invalid. No inference repeated. Counters20974363 failed after OOM/native−4 on requeue; its partial record is preserved. Two jobs20863185 and20974364 were requeued at06:03; their reset elapsed times are not total campaign compute.

All19 Rover evaluations are terminal and have matching VAL summaries including OOM-labelled allocations. These are fixed-budget results; interrupted work is never silently counted as successful.


## Stage-2 MCTS — every domain and both branches
| Cell | Branch | Search | n terminal | Policy | MCTS30m/2h/6h | Δ [95% CI] | raw p | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_grouping/off | validation_led | narrow5/20 | 10 | 16 | 11.4 / 15 / 15.7 | -0.3 [-0.889, 0.289] | 0.5 | no reliable gain or loss at6h;30m is materially worse |
| block_grouping/on | validation_led | narrow5/20 | 10 | 12.8 | 10.1 / 10.6 / 12.6 | -0.2 [-1.08, 0.679] | 0.812 | no reliable gain or loss at6h;30m is materially worse |
| block_grouping/off | terminal_led | narrow5/20 | 10 | 16.3 | 4.4 / 12.3 / 14.5 | -1.8 [-3.22, -0.38] | 0.0312 | MCTS loses coverage |
| block_grouping/on | terminal_led | narrow5/20 | 10 | 11.6 | 4.4 / 8.7 / 9.4 | -2.2 [-4.6, 0.2] | 0.0312 | MCTS loses mean coverage |
| drone/off | validation_led | normal20/70 | 10 | 6.7 | 7.5 / 7.7 / 7.7 | 1 [-0.43, 2.43] | 0.195 | positive mean but not significant |
| drone/on | validation_led | normal20/70 | 10 | 5 | 10.9 / 11.2 / 11.2 | 6.2 [4.33, 8.07] | 0.00195 | large significant gain |
| drone/off | terminal_led | normal20/70 | 10 | 7.8 | 9.7 / 9.8 / 9.9 | 2.1 [0.65, 3.55] | 0.0195 | positive paired effect but not Holm-significant |
| drone/on | terminal_led | normal20/70 | 10 | 6.5 | 12.9 / 13.1 / 13.1 | 6.6 [5.47, 7.73] | 0.00195 | large significant gain |
| fo_counters/off | validation_led | normal20/70 | 7 | 2.9 | ≥ / ≥ / ≥ | live |  | wait for four live seeds |
| fo_counters/on | validation_led | normal20/70 | 9 | 3.1 | ≥ / ≥ / ≥ | live |  | wait for one live seed |
| fo_counters/off | terminal_led | normal20/70 | 10 | 2.8 | 5.3 / 5.3 / 5.3 | 2.5 [1.73, 3.27] | 0.00195 | strong significant gain; job20559567 retry duplicates deduplicated to four unique instances and all four are VAL-valid |
| fo_counters/on | terminal_led | normal20/70 | 8 | 3.8 | ≥4.4 / ≥4.4 / ≥4.4 | live |  | current all-ten lower bound is+0.6; wait for two live seeds |
| rover/off | validation_led | normal20/70 | 10 | 4 | 4.5 / 4.5 / 4.5 | 0.5 [-0.00583, 1.01] | 0.125 | Positive mean; exact paired test determines evidence |
| rover/on | validation_led | normal20/70 | 10 | 3.9 | 4.4 / 4.5 / 4.5 | 0.6 [-0.0911, 1.29] | 0.156 | Positive mean; exact paired test determines evidence |
| rover/off | terminal_led | normal20/70 | 10 | 3.8 | 4.2 / 4.2 / 4.2 | 0.4 [0.03, 0.77] | 0.125 | small non-significant gain |
| rover/on | terminal_led | normal20/70 | 10 | 4 | 4.5 / 4.5 / 4.5 | 0.5 [0.12, 0.88] | 0.0625 | small non-significant gain |
| counters/off | validation_led | narrow5/20 | 10 | 36.9 | 34.9 / 36.7 / 36.7 | -0.2 [-2.87, 2.47] | 0.969 | no meaningful change |
| counters/on | validation_led | narrow5/20 | 10 | 21.8 | 22.6 / 26.4 / 27.1 | 5.3 [-0.7, 11.3] | 0.082 | positive mean but high variance |
| counters/off | terminal_led | narrow5/20 | 4 | 37.9 | ≥34.9 / ≥37.8 / ≥38.4 | live |  | ten seeds newly running |
| counters/on | terminal_led | narrow5/20 | 3 | 16.8 | ≥19.9 / ≥22.0 / ≥22.4 | live |  | ten seeds newly running |


## PW20 and PW70 — identical two-seed cells
| domain | stage | value_head | n | policy_mean | fixed_search | fixed_30m | fixed_2h | fixed_6h | pw20_30m | pw20_2h | pw20_6h | pw70_30m | pw70_2h | pw70_6h | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_grouping | stage1 | off | 2 | 16.5 | fixed_narrow_5_width_20_sim | 11.0 | 13.5 | 15.0 | 11.5 | 15.0 | 15.0 | 10.5 | 11.5 | 13.0 | terminal |
| block_grouping | stage1 | on | 2 | 17.0 | fixed_narrow_5_width_20_sim | 11.5 | 13.5 | 17.0 | 12.5 | 17.0 | 18.0 | 9.0 | 11.5 | 13.5 | terminal |
| counters | stage1 | off | 2 | 18.0 | fixed_narrow_5_width_20_sim | 21.5 | 21.5 | 21.5 | 20.5 | 20.5 | 20.5 | >=21.0 | >=21.0 | >=21.0 | live |
| counters | stage1 | on | 2 | 5.0 | fixed_narrow_5_width_20_sim | 13.5 | 13.5 | 13.5 | 12.5 | 12.5 | 12.5 | 12.0 | 12.5 | 12.5 | terminal |
| counters | stage2 | off | 2 | 49.0 | fixed_narrow_5_width_20_sim | >=37.0 | >=41.5 | 44.5 | 46.0 | 49.5 | 49.5 | 34.0 | 39.5 | 43.5 | terminal |
| counters | stage2 | on | 2 | 5.0 | fixed_narrow_5_width_20_sim | 16.5 | 17.5 | 17.5 | 15.0 | 15.0 | 15.0 | 13.5 | 14.0 | 14.0 | terminal |
| fo_counters | stage1 | off | 2 | 3.5 | fixed_normal_20_width_70_sim | 9.0 | 9.5 | 9.5 | NA | NA | NA | 9.5 | 9.5 | 9.5 | terminal_val_confirmed |
| fo_counters | stage1 | on | 2 | 3.5 | fixed_normal_20_width_70_sim | 6.0 | 7.5 | 7.5 | NA | NA | NA | 8.0 | 8.0 | 8.0 | terminal_val_confirmed |
| rover | stage1 | off | 2 | 4.0 | fixed_normal_20_width_70_sim | 4.0 | 4.5 | 4.5 | NA | NA | NA | 5.0 | 5.0 | 5.0 | terminal_val_confirmed |
| rover | stage1 | on | 2 | 4.0 | fixed_normal_20_width_70_sim | 4.5 | 4.5 | 4.5 | NA | NA | NA | 5.5 | 5.5 | 5.5 | terminal_val_confirmed |


Correction: the5 September report used PW20 values in several fixed-comparator columns. The underlying distinct columns are preserved and used here. No interpretation should rely on the swapped columns.


## PW70 five-seed conclusions
| domain | stage | value_head | policy_mean | fixed_30m | fixed_2h | fixed_6h | pw70_30m | pw70_2h | pw70_6h | pw_vs_policy_6h | ci95_low | ci95_high | raw_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| counters | stage2 | off | 37.8 | 34.6 | 36.4 | 36.4 | 29.4 | 33.6 | 36.4 | -1.4 | -9.757053329523899 | 6.957053329523898 | 0.9375 |
| counters | stage2 | on | 32.4 | 27.4 | 34.2 | 35.6 | 22.0 | 28.4 | 31.4 | -1.0 | -13.197413003333224 | 11.197413003333224 | 0.875 |
| fo_counters | stage1 | off | 3.6 | 8.0 | 8.6 | 8.6 | 8.0 | 8.0 | 8.0 | 4.4 | 2.52 | 6.28 | 0.0625 |
| fo_counters | stage1 | on | 3.2 | 4.8 | 5.4 | 5.4 | 7.2 | 7.2 | 7.2 | 4.0 | 2.76 | 5.24 | 0.0625 |
| rover | stage1 | off | 4.0 | 4.6 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 0.8 | -0.56 | 2.16 | 0.5 |
| rover | stage1 | on | 3.6 | 4.6 | 4.6 | 4.6 | 4.8 | 4.8 | 4.8 | 1.2 | -0.84 | 3.24 | 0.5 |


FO/Rover are being expanded to10; Counters is not expanded. Five was a resource-saving screen and could not reach two-sided exactp<.05. Ten does not guarantee significance. Post-hoc30m/2h is defensible for observed time-to-solution coverage; it does not estimate fresh whole-job runtime or remove OOM/censoring.


## MPrime validation Phase B
240 independent candidates across two pools and three structural tiers generated with frozen seeds/hashes. Goal locations are separated from all initial pleasure positions by graph distance; direct two-action witnesses are not inserted. ENHSP and VAL screen solvability before any network score is read. Each pool selects the first10 certified candidates per tier in seed order. This is a deliberately harder validation hypothesis, not proof of adequacy.

The complete inventory contains1130 saved checkpoints across60 lineages. Scoring both independent sets means2260 checkpoint-replicate evaluations, not1130. PhaseB uses existing weights. The60-lineage campaign assesses checkpoint selection; a separate follow-up rescore of the28 anchor candidates is needed to test anchor-rank stability. No MPrime MCTS submitted.


## Every registered experiment, including held and historical work
| experiment_id | status | scope | next_action | results_file |
| --- | --- | --- | --- | --- |
| MAIN-VAL | completed | 5 domains x 2 VH x 10 seeds | Policy experiment complete at 300/300 endpoints; do not routinely reprint unless requested | experiment_tracking/experiment_results.csv |
| MAIN-TERM | completed | 5 domains x 2 VH x 10 seeds | Policy experiment complete; do not routinely reprint unless requested | experiment_tracking/experiment_results.csv |
| PRESERVE-3-VAL | completed | Delivery/TPP/Zenotravel x 2 VH x 10 seeds | Zenotravel is preserved and Delivery is essentially preserved; TPP is not fully preserved because of one audited9/20 off-mode collapse | experiment_tracking/four_domain_preservation/stable_domain_stage2_summary_20260902.csv |
| PRESERVE-3-TERM | completed | Delivery/TPP/Zenotravel x two VH x ten final Stage-1 checkpoints | All 60 training lineages and all selected endpoints complete; all nineteen final TPP curve repair jobs completed; freeze as completed. | experiment_tracking/preserve3_terminal_led/terminal_led_selected_summary_20260903.csv |
| MPRIME-VAL | completed-reclassified | MPrime validation selection and corrected Stage-1 replication | Corrected Stage1 and all 290 policy jobs are terminal; validation improves selected versus final means but pooled Spearman agreement is about 0.25; preservation failed so MPrime moves to the six-domain imperfect extension | experiment_tracking/mprime_validation_ipc_scale_v1/validation_test_checkpoint_audit.csv |
| MAIN-EXT6-MPRIME | completed-policy-with-validation-audit | MPrime x two VH x ten seeds after corrected Stage-1 selection | Retain results as provisional extension and complete MPRIME-VAL-ADEQUACY before further MPrime training | experiment_tracking/mprime_validation_ipc_scale_v1/validation_led_stage2_selected_statistics_20260903.csv |
| MAIN-TERM-EXT6-MPRIME | completed-policy | MPrime x two VH x ten corrected Stage-1 final checkpoints | All twenty training and420 policy evaluations terminal; selected epoch0 means14.0 off and14.5 on and neither paired change is significant | experiment_tracking/mprime_validation_ipc_scale_v1/terminal_led_stage2_selected_statistics_20260902.csv |
| ANCHOR-4 | completed | Delivery/TPP/Zenotravel frozen; MPrime anchor10 frozen both VH from588/588 points | MPrime anchor10 wins both VH; validation-stability audit remains separate | experiment_tracking/mprime_validation_ipc_scale_v1/anchor_selection_corrected_frozen_20260901.csv |
| MCTS-WIDTH | completed | Block Grouping and Counters primary corrections | Counters Stage1 and Stage2 are terminal; Stage2 off is 36.7/59 at 2h/6h versus policy 36.9 and on is 26.4/59 at 2h or 27.1/59 at 6h versus policy 21.8; post-hoc VAL jobs 20768679--20768681 found zero invalid plans | experiment_tracking/mcts_counters_width_sensitivity/stage2_narrow_summary_10seed.csv |
| MCTS-PW | completed | Drone x 2 VH x 5 seeds | 30/30 terminal and VAL-valid; PW is significantly faster and retains far fewer nodes but significantly loses coverage | experiment_tracking/mcts_progressive_widening_pilot/summary.csv |
| MCTS-SAFE | completed-diagnostic | Four targeted Drone policy-only-loss instances | SAFE-1 is useful but incomplete; MCTS-SAFE-CONTEXT and MCTS-SAFE-2 are separately documented follow-ups | experiment_tracking/mcts_safe_drone/targeted_results_20260828.csv |
| MCTS-SAFE2 | held-low-priority | Future matched horizon-binding instances | Activate only after nonzero cutoffs and measurable cross-horizon state reuse establish a contamination mechanism |  |
| MCTS-SAFE-CONTEXT | completed-negative | Four-domain three-instance diagnostics plus Drone matched physical/contextual evaluation | Ten matched pairs are terminal; contextual nodes reduce VH-off by 0.4 and VH-on by 3.4 plans; neither VH result survives Holm correction; retain diagnostics but do not promote behavior | experiment_tracking/mcts_safe_context/live_reconciliation_20260831.csv |
| MCTS-HORIZON | completed-nonresult | Drone x 2 VH x 5 matched seeds x aware/unaware | All twenty arms and post-hoc VAL are complete; paired mean change is zero with 95 percent CI -0.89 to 0.89 and sign-flip p 1.0; almost no cutoffs bound so Counters is the proper efficacy follow-up | experiment_tracking/mcts_horizon_binding/results.csv |
| MCTS-HORIZON-COUNTERS | completed-negative | Counters x two VH x two matched seeds x aware/unaware | Freeze as a non-result: both VH modes have zero aware-minus-unaware change at6h and zero recorded cutoffs. | experiment_tracking/status_20260905_latest.md |
| MCTS-PW-SAFE | extension-completed | Initial Drone two-seed four-arm gate plus Kmin3 extension on four additional seeds in both VH modes | All eight Kmin3 jobs are terminal; Kmin3 averages 9.5 versus policy 7.0 and top20 10.5; freeze rather than silently expanding | experiment_tracking/mcts_progressive_widening_sensitivity/results.csv |
| MCTS-PW-CROSS-DOMAIN | completed-screen | Block Grouping FO Counters Rover and Counters with two seeds and both VH modes | 20/20 terminal and printed plans VAL-confirmed; never label the whole screen PW20 because the eight FO Counters/Rover rows use PW70 | experiment_tracking/mcts_progressive_widening_cross_domain/comparative_summary_20260901_1022.csv |
| MCTS-PW70-CROSS-DOMAIN | live-one-job-tail | Block Grouping Stage1 and Counters Stage1/validation-led Stage2 with two seeds and both VH modes | One Counters Stage1/off seed remains running; retain >= lower bounds until terminal. | experiment_tracking/mcts_progressive_widening_cross_domain/pw70_live_summary_20260905.csv |
| MCTS-PW70-CONFIRMATORY | completed-five-seed-screen | Counters validation-led Stage2 FO Counters Stage1 and Rover Stage1 x both VH x five matched seeds | All five seed cells terminal; expand FO/Rover to ten with separately registered20 jobs; Counters PW70 does not match policy consistently. | experiment_tracking/mcts_progressive_widening_cross_domain/pw70_live_summary_20260905.csv |
| MCTS-PW-30M | held-optional-execution-validation | Promising cells selected from cross-domain screening | Post-hoc cutoffs suffice for solutions-within-budget coverage; fresh runs needed only for whole-job resource/time validation or missing/censored timing evidence. |  |
| MAIN-VAL-S2-MCTS | completed | Drone x two VH x ten seeds | All twenty exact endpoints terminal with VAL evidence | experiment_tracking/main_val_stage2_drone_mcts_summary_20260901.csv |
| MCTS-LEGACY-ROVER | completed | 20 terminal-led plus one validation-led Rover Stage2 endpoint | Off policy3.8 to MCTS4.2 CI[0.03 0.77] rawp.125; on policy4.0 to MCTS4.5 CI[0.12 0.88] rawp.0625; OOM allocations retained as fixed-budget outcomes | experiment_tracking/rover_stage2_terminal_mcts_20260901.csv |
| MCTS-LEGACY-FO | live-two-job-tail | 20 terminal-led FO Counters Stage2 endpoints | VH-off complete and significant; two VH-on seeds remain running; deduplicate and VAL-certify job20559567 without repeating inference. | experiment_tracking/stage2_policy_mcts_comparison_by_branch_20260905.csv |
| MCTS-STAGE2-BRANCH-COMPLETION | 40-terminal-17-running | All57 identities submitted; BG13 terminal Rover19 terminal FO1 new terminal; Counters7 terminal including1 failed partial | FO4 and Counters13 run; include original FO terminal2 separately. Failed Counters20974363 has22 VAL-valid partial successes and needs infrastructure-aware remaining-instance recovery. | experiment_tracking/stage2_policy_mcts_comparison_by_branch_20260905.csv |
| MCTS-PW-PATHBATCH | held-design | Future matched Drone pilot after standard PW sensitivity | Freeze update semantics and compare coverage runtime generated states batch size retained nodes and memory |  |
| MCTS-RESOURCE | held | FO Counters/Rover interrupted endpoints | Deploy tested lifecycle commit, then resume |  |
| LONG-DRONE | completed-policy-held-mcts | 3 scheduler-limited Drone S1 lineages | All six policy endpoints and three final-checkpoint MCTS endpoints are complete; release the three selected-checkpoint MCTS only if this side comparison is prioritized | experiment_tracking/long_drone_endpoint_results.csv |
| STOP-ORIG | held | Stage-1 replication | Finalize compatibility implementation and launch manifest |  |
| PUCT-EST | held | Block Grouping historical reproduction | Do not submit until released |  |
| ENHSP-LEAF | completed | Selected domains/endpoints | Archive final logs/results | experiment_tracking/experiment_results.csv |
| BG-HIST | completed | One historical checkpoint/configuration family | Archive provenance; do not use width-5/3 as primary | experiment_tracking/experiment_results.csv |
| MCTS-DETERMINISM-AUDIT | completed-diagnostic | Two six-repeat node-family audits plus fresh same-node aware/unaware replay | CPU-family numerical differences change internal checksums but no observed action; aware/unaware replay is identical with zero cutoffs | experiment_tracking/mcts_determinism_audit/followup_results.csv |
| ACT-HISTORY-ABLATION | held-design | Fresh Stage1 training on Drone and Counters plus mostly-monotone TPP control | Keep held; TPP is the control because bought is monotone nondecreasing and on-sale monotone nonincreasing; only drive location can cycle | experiment_tracking/act_history_ablation.md |
| MCTS-PW-COUNTERS-DIVERGENCE | pw20-complete-pw70-one-terminal-two-live | Three exact Stage1 VH-off snapshots with severe policy-to-fixed-narrow regressions x PW20/PW70 | PW70 seed923500475 TIMEOUT21/59; other2 live. One requeued at06:03; preserve total budget/restart provenance. | experiment_tracking/mcts_progressive_widening_cross_domain/pw70_live_summary_20260905.csv |
| ANCHOR-KL-CONTROL | held-ready | TPP/off outlier plus stable matched seed and Block Grouping VH-on diagnostic seeds | Keep held; implement and smoke-test the generic coefficient controller before the12-job screening campaign | experiment_tracking/anchor_kl_control_literature_and_design_20260902.md |
| MPRIME-VAL-ADEQUACY | phase-b-planner-screen-live | All corrected MPrime Stage1 and both Stage2 branches | 240 candidates created; planner array21039224; afterok finalizer21039342 freezes60 instances and schedules preflight then60 rescore lineages; no MPrime MCTS authorized. | experiment_tracking/mprime_validation_ipc_scale_v1/validation_adequacy_phase_a_stage2_summary_20260903.csv |
| MCTS-PW70-TEN-SEED | submitted | FO Counters and Rover S1,2 VH,5 new seeds per cell;20 jobs | Jobs21039201-21039220; reuse the existing five seeds; do not select subsets by outcomes. | experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_submissions_20260906.tsv |


## Completed RQs and experiments — static results retained
## Research questions

### RQ1 — does MCTS-guided continued training improve VH-off policy?

| Domain | Validation-led change [95% CI]; Holm p | Terminal-led change [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -0.3 [-1.20, 0.60]; 1.000 | 0.0 [-0.95, 0.95]; 1.000 | No improvement |
| Drone | +0.8 [-1.27, 2.87]; 1.000 | +0.4 [-1.33, 2.13]; 1.000 | No reliable improvement |
| FO Counters | -1.3 [-2.37, -0.23]; .234 | -0.8 [-1.46, -0.14]; .219 | Negative raw effect; not Holm-significant |
| Rover | 0.0 [0, 0]; 1.000 | 0.0 [-0.34, 0.34]; 1.000 | No change |
| Counters | +4.4 [-17.44, 26.24]; 1.000 | +16.4 [3.09, 29.71]; .137 | Large noisy mean; not Holm-significant |

Conclusion: no original five-domain RQ1 cell survives within-RQ Holm correction.

### RQ3 — does the value head improve refinement?

The estimand is a paired difference-in-differences: `(VH-on S2−S1) −
(VH-off S2−S1)`.

| Domain | Validation-led DiD [95% CI]; Holm p | Terminal-led DiD [95% CI]; Holm p | Conclusion |
|---|---|---|---|
| BG | -2.8 [-4.61, -0.99]; .088 | -3.4 [-5.13, -1.67]; .020 | Terminal-led VH significantly worsens refinement |
| Drone | -0.9 [-2.88, 1.08]; 1.000 | -1.1 [-3.57, 1.37]; .836 | No reliable effect |
| FO Counters | +0.7 [-0.26, 1.66]; .813 | +1.3 [0.08, 2.52]; .188 | Positive raw tendency only |
| Rover | +0.1 [-0.31, 0.51]; 1.000 | -0.1 [-0.51, 0.31]; 1.000 | No effect |
| Counters | -1.2 [-25.87, 23.47]; 1.000 | -16.8 [-29.64, -3.96]; .078 | Large negative tendency; high variance |

Conclusion: only terminal-led Block Grouping is significant after correction.

### RQ2/RQ4 — Stage-2 inference-time MCTS
The updated all-domain, both-branch table above supplies the current evidence. Rover validation is now complete. FO validation/on-terminal and Counters terminal remain live.

## Completed experiments and static conclusions

### Stage-1 selected policy versus preferred MCTS

| Domain/VH | Search | Policy | MCTS 30m / 2h / 6h | 6h change [95% CI] | Raw / Holm p | Conclusion |
|---|---|---:|---:|---|---:|---|
| BG/off | narrow 5/20 | 16.3 | 11.6 / 14.8 / 15.4 | -0.9 [-1.88, 0.08] | .109 / .547 | No reliable benefit |
| BG/on | narrow 5/20 | 15.9 | 12.0 / 14.0 / 16.2 | +0.3 [-0.29, 0.89] | .453 / .750 | No reliable benefit |
| Drone/off | normal 20/70 | 5.9 | 6.9 / 6.9 / 6.9 | +1.0 [0.05, 1.95] | .074 / .445 | Positive but not corrected-significant |
| Drone/on | normal 20/70 | 5.1 | 10.0 / 10.4 / 10.4 | +5.3 [3.42, 7.18] | .002 / .020 | Significant gain |
| FO/off | normal 20/70 | 4.2 | 7.5 / 7.8 / 7.8 | +3.6 [2.20, 5.00] | .004 / .035 | Significant gain |
| FO/on | normal 20/70 | 3.7 | 5.3 / 5.7 / 5.7 | +2.0 [1.05, 2.95] | .004 / .035 | Significant gain |
| Rover/off | normal 20/70 | 4.0 | 4.8 / 5.0 / 5.0 | +1.0 [0.25, 1.75] | .031 / .219 | Positive; not corrected-significant |
| Rover/on | normal 20/70 | 3.8 | 4.4 / 4.4 / 4.4 | +0.6 [-0.00, 1.20] | .125 / .547 | No reliable effect |
| Counters/off | narrow 5/20 | 32.5 | 24.9 / 25.6 / 25.7 | -6.8 [-17.73, 4.13] | .250 / .750 | Search regression on average |
| Counters/on | narrow 5/20 | 18.6 | 20.3 / 22.1 / 22.5 | +3.9 [-4.33, 12.13] | .328 / .750 | Positive noisy mean |

### PRESERVE-3 validation-led

| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -0.2 [-1.01, .61] | 1.0 | Essentially preserved |
| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +0.4 [-.37, 1.17] | .5 | Preserved |
| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59, 1.39] | 1.0 | Not uniformly preserved |
| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -0.5 [-1.63, .63] | 1.0 | Essentially preserved |
| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0, 0] | 1.0 | Preserved |
| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -0.1 [-.33, .13] | 1.0 | Preserved |

`*` Nine TPP/off seeds score 20/20; seed 1972442430 scores 9/20 after a
seed-specific Stage-2 collapse. It is not a broad mild decline.

### PRESERVE-3 terminal-led

| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---:|---:|---|---:|---|
| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52, 5.12] | .496 | Preserved relative to final source |
| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-.91, 4.31] | .188 | Preserved/improved mean |
| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +0.4 [-2.13, 2.93] | 1.0 | Preserved on average |
| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | -0.6 [-3.77, 2.57] | 1.0 | No reliable change |
| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | -0.2 [-.50, .10] | .5 | Essentially preserved |
| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +0.2 [-.10, .50] | .5 | Preserved |

All 60 terminal-led lineages, all selected endpoints and the nineteen final TPP
curve-repair jobs are now Slurm-complete. No PRESERVE-3 compute remains live.

### MPrime policy extension and validation adequacy

| Branch/VH | Stage-1 | Stage-2 selected | Change [95% CI] | Raw p | Conclusion |
|---|---:|---:|---|---:|---|
| Validation/off | 15.0 | 15.2 | +0.2 [-1.18, 1.58] | .880 | Provisional neutral |
| Validation/on | 14.6 | 15.2 | +0.6 [-1.46, 2.66] | .580 | Provisional neutral |
| Terminal/off | 13.5 | 14.0 | +0.5 [-1.29, 2.29] | .672 | Provisional neutral |
| Terminal/on | 13.2 | 14.5 | +1.3 [-.49, 3.09] | .172 | Positive tendency only |

These policy results remain selector-provisional; Phase B is now running as described above.

### Other completed side experiments

| Experiment | Conclusion | Provenance |
|---|---|---|
| Counters binding Horizon | Both VH modes have aware−unaware 6h change 0.0 [0,0], p=1.0; no benefit and zero logged cutoffs | `mcts_horizon_binding/horizon_completion_records_20260902.csv` plus cluster completion records |
| Drone Horizon | Same-commit aware/unaware change 0.0 [-.89,.89], p=1.0 | `mcts_horizon_determinism_causal/` |
| MCTS-SAFE-1 | Repaired 2/4 targeted Drone dead-end outcomes; useful safety guard but not a complete search-quality fix | `mcts_safe_preflight/` |
| MCTS-SAFE-CONTEXT | Negative: off -0.4 [-1.08,.28], p=.5; on -3.4 [-6.64,-.16], p=.0625 | `mcts_safe_context/live_reconciliation_20260831.csv` |
| Determinism audits | CPU-family numerical checksums differ, but audited actions and outcomes did not; timing randomness is unsupported | `mcts_determinism_audit/README.md` |



## Long Drone detailed endpoints
| value_head | seed | checkpoint_role | cumulative_epoch | policy_score | mcts_score | notes |
| --- | --- | --- | --- | --- | --- | --- |
| on | 534933607 | continuation_selected | 243 | 11 |  | Continuation reset validation baseline incorrectly; continuation-selected policy equals final policy at 11/20 and never exceeded the original lineage best |
| on | 534933607 | final | 293 | 11 | 14 | Final endpoint policy and MCTS complete |
| off | 1963100312 | continuation_selected | 250 | 13 |  | Continuation-selected policy endpoint complete; selected-checkpoint MCTS remains held |
| off | 1963100312 | final | 300 | 9 | 9 | Final endpoint policy and MCTS complete |
| off | 1472491096 | continuation_selected | 154 | 13 |  | Continuation-selected policy endpoint complete; selected-checkpoint MCTS remains held |
| off | 1472491096 | final | 204 | 13 | 13 | Final endpoint policy and MCTS complete |


## Provenance
Every current live row references its SlurmID and log. New comparisons reference row-level companions containing checkpoint, training log, policy log and MCTS log. [rover_validation_seed_results_20260906.csv](C:/Users/roeeh/Desktop/School/Meitar/Thesis/Code/numeric-asnets/.codex-worktrees/mcts-safe-context/experiment_tracking/rover_validation_seed_results_20260906.csv); [job_refresh_20260906.csv](C:/Users/roeeh/Desktop/School/Meitar/Thesis/Code/numeric-asnets/.codex-worktrees/mcts-safe-context/experiment_tracking/job_refresh_20260906.csv); [snapshot_provenance_index_20260905.csv](C:/Users/roeeh/Desktop/School/Meitar/Thesis/Code/numeric-asnets/.codex-worktrees/mcts-safe-context/experiment_tracking/snapshot_provenance_index_20260905.csv). Older dated records remain unchanged.
