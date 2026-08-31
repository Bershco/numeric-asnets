#!/usr/bin/env bash
set -euo pipefail

root=/home/hersco/training_new_domains/2026-08-31/preserve3_terminal_stage2
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
controller="$repo/scripts/stage2_policy_refresh.sbatch"

for domain in delivery tpp zenotravel; do
  for value_head in off on; do
    cell="${domain}_${value_head}"
    ledger="$root/terminal_stage2_${cell}.tsv"
    [[ -s "$ledger" ]] || { echo "missing ledger: $ledger" >&2; exit 2; }

    mapfile -t job_ids < <(awk -F '\t' 'NR > 1 {gsub(/\r/, "", $12); print $12}' "$ledger")
    [[ "${#job_ids[@]}" -eq 10 ]] || {
      echo "$cell has ${#job_ids[@]} training jobs; expected 10" >&2
      exit 2
    }
    dependency=$(IFS=:; echo "${job_ids[*]}")
    ready="$root/policy_ready_${cell}.csv"
    submitted="$root/policy_submissions_${cell}.csv"
    output_prefix="$root/policy_eval/${cell}"

    sbatch --parsable --dependency="afterany:${dependency}" \
      --job-name="P3_${domain^^}_${value_head^^}_POLICY" \
      --export="ALL,TRAINING_LEDGER=$ledger,READY_MANIFEST=$ready,POLICY_LEDGER=$submitted,ROLE_PREFIX=preserve3_term_${cell},SUFFIX_PREFIX=P3T_${cell},OUTPUT_PREFIX=$output_prefix" \
      "$controller"
  done
done
