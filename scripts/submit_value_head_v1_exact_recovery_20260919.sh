#!/bin/bash
set -euo pipefail
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
EXCLUDE='ise-cpu128-03,ise-cpu-intl-[01,09-15,25-28]'

submit_capture() {
  local domain=$1 array=$2 sources=$3 mem=$4 limit=$5 name=$6
  local raw
  raw=$(sbatch --parsable --array="$array" --mem="$mem" --time="$limit" --exclude="$EXCLUDE" \
    --job-name="$name" \
    --export="ALL,DOMAIN=$domain,CAPTURE_SOURCES=$sources,MATERIALIZE_AFTER_CAPTURE=0" \
    "$REPO/scripts/value_head_v1_state_capture.sbatch")
  printf '%s\n' "${raw%%;*}"
}

FO_RANDOM=$(submit_capture fo_counters 0-1 common_random_legal 96G 04:00:00 vhv1fix_fo_random)
FO_STAGE2=$(submit_capture fo_counters 1 stage2_on_policy 96G 04:00:00 vhv1fix_fo_s2)
ROVER_STAGE2=$(submit_capture rover 0 stage2_on_policy 96G 04:00:00 vhv1fix_rover_s2)
MPRIME=$(submit_capture mprime 0-1 common_random_legal,stage1_on_policy,stage2_on_policy 120G 08:00:00 vhv1fix_mprime)

PARENTS="$FO_RANDOM:$FO_STAGE2:$ROVER_STAGE2:$MPRIME"
MANIFEST=$(sbatch --parsable --array=0-7 --dependency="afterok:$PARENTS" \
  "$REPO/scripts/value_head_v1_manifest_rebuild.sbatch")
MANIFEST=${MANIFEST%%;*}
FINAL=$(sbatch --parsable --dependency="afterok:$MANIFEST" \
  "$REPO/scripts/value_head_v1_state_finalize.sbatch")
FINAL=${FINAL%%;*}
POST=$(sbatch --parsable --dependency="afterok:$FINAL" \
  "$REPO/scripts/value_head_v1_postcapture_controller.sbatch")
POST=${POST%%;*}

{
  printf 'role\tjob_id\tdependency\n'
  printf 'fo_random\t%s\t\n' "$FO_RANDOM"
  printf 'fo_stage2_seed923500475\t%s\t\n' "$FO_STAGE2"
  printf 'rover_stage2_seed534933607\t%s\t\n' "$ROVER_STAGE2"
  printf 'mprime_random_and_policies\t%s\t\n' "$MPRIME"
  printf 'manifest_rebuild\t%s\tafterok:%s\n' "$MANIFEST" "$PARENTS"
  printf 'state_finalizer\t%s\tafterok:%s\n' "$FINAL" "$MANIFEST"
  printf 'postcapture_controller\t%s\tafterok:%s\n' "$POST" "$FINAL"
} > "$ROOT/v1_exact_recovery_20260919.tsv"

printf 'V1_EXACT_RECOVERY|fo_random=%s|fo_s2=%s|rover_s2=%s|mprime=%s|manifest=%s|final=%s|post=%s\n' \
  "$FO_RANDOM" "$FO_STAGE2" "$ROVER_STAGE2" "$MPRIME" "$MANIFEST" "$FINAL" "$POST"
