#!/usr/bin/env bash
#SBATCH --job-name=val-fo-20559567-dedup
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-05/fo_job20559567_dedup_val_%j.out

set -euo pipefail

checkout=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
source_log=/home/hersco/training_new_domains/2026-08-25/statistical_replication_terminal_stage2_mcts_eval/20559567_Ev_fo_counters_fo_counters_mcts_orig_novh_e.5_c.1_s1510771779_K0_SR10TCM_src20489451_e0009.txt
summary=/home/hersco/training_new_domains/2026-09-05/fo_job20559567_dedup_val_summary.csv

python3 "$checkout/asnets/tools/validate_eval_log_with_summary.py" \
  --log "$source_log" \
  --domain fo_counters \
  --validator /home/hersco/tools/VAL/build/bin/Validate \
  --asnets-dir "$checkout/asnets" \
  --allow-incomplete \
  --summary-csv "$summary"
