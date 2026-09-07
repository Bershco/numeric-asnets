#!/usr/bin/env bash
set -euo pipefail
while IFS='|' read -r job log; do
  printf 'FILE|%s|%s\n' "$job" "$log"
  if [[ ! -f "$log" ]]; then printf 'MISSING|%s\n' "$job"; continue; fi
  tail -c 5000000 "$log" | grep -aE '\[EVAL FINAL\]|VAL summary|Plan validation|timed out after|TIMEOUT|OUT_OF_MEMORY|success=' | tail -n 25 || true
done <<'ROWS'
20945846|/home/hersco/training_new_domains/2026-09-04/statistical_replication_stage2_mcts_eval/20945846_Ev_fo_counters_fo_counters_mcts_orig_vh_e.5_c.1_s2011206605_K0_SR10M_src20430378_e0046.txt
20755796|/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/20755796_pw70-kmin3-counters-off-2011206605-s1.txt
20974362|/home/hersco/training_new_domains/2026-09-05/statistical_replication_stage2_mcts_eval/20974362_Ev_counters_counters_mcts_orig_vh_e.5_c.1_s1510771779_K0_SR10M_src20489420_e0001.txt
20559575|/home/hersco/training_new_domains/2026-08-25/statistical_replication_terminal_stage2_mcts_eval/20559575_Ev_fo_counters_fo_counters_mcts_orig_vh_e.5_c.1_s1073581256_K0_SR10TCM_src20489458_e0077.txt
20974542|/home/hersco/training_new_domains/2026-09-05/statistical_replication_stage2_mcts_eval/20974542_Ev_counters_counters_mcts_orig_novh_e.5_c.1_s1510771779_K0_SR10M_src20489410_e0045.txt
20974546|/home/hersco/training_new_domains/2026-09-05/statistical_replication_stage2_mcts_eval/20974546_Ev_counters_counters_mcts_orig_vh_e.5_c.1_s923500475_K0_SR10M_src20489416_e0030.txt
20974540|/home/hersco/training_new_domains/2026-09-05/statistical_replication_stage2_mcts_eval/20974540_Ev_counters_counters_mcts_orig_novh_e.5_c.1_s1472491096_K0_SR10M_src20489409_e0002.txt
20974354|/home/hersco/training_new_domains/2026-09-05/statistical_replication_stage2_mcts_eval/20974354_Ev_counters_counters_mcts_orig_novh_e.5_c.1_s1972442430_K0_SR10M_src20489412_e0094.txt
20974361|/home/hersco/training_new_domains/2026-09-05/statistical_replication_stage2_mcts_eval/20974361_Ev_counters_counters_mcts_orig_vh_e.5_c.1_s1472491096_K0_SR10M_src20489419_e0070.txt
20974346|/home/hersco/training_new_domains/2026-09-05/statistical_replication_stage2_mcts_eval/20974346_Ev_counters_counters_mcts_orig_novh_e.5_c.1_s534933607_K0_SR10M_src20489405_e0036.txt
21039220|/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/21039220_pw70-ten-rover-on-2082152039-s1.txt
21039215|/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/21039215_pw70-ten-rover-off-2082152039-s1.txt
21039209|/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/21039209_pw70-ten-fo_counters-on-1972442430-s1.txt
21039205|/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/21039205_pw70-ten-fo_counters-off-2082152039-s1.txt
20863185|/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/20863185_pw-counters-divergence-off-534933607-sim70.txt
ROWS
