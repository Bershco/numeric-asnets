#!/bin/bash
set -euo pipefail

REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
REQUIRED_FIX=7ebc2c16a7a3d7355e3a52effec3409ce5f9dfb4
EXCLUDE='ise-cpu128-03,ise-cpu-intl-[01,09-15,25,27-28]'

git -C "$REPO" merge-base --is-ancestor "$REQUIRED_FIX" HEAD
test -e "$REPO/asnets/asnets/ops/_asnet_ops_impl.so"

submit() {
  local raw
  raw=$(sbatch --parsable --exclude="$EXCLUDE" "$@")
  printf '%s\n' "${raw%%;*}"
}

smoke=$(submit "$REPO/scripts/value_head_v1_state_capture_smoke.sbatch")
drone=$(submit --dependency="afterok:$smoke" --array=0-1 --mem=48G --time=08:00:00 \
  --job-name=vhv1cap_drone --export=ALL,DOMAIN=drone \
  "$REPO/scripts/value_head_v1_state_capture.sbatch")
fo=$(submit --dependency="afterok:$smoke" --array=0-1 --mem=64G --time=08:00:00 \
  --job-name=vhv1cap_fo --export=ALL,DOMAIN=fo_counters \
  "$REPO/scripts/value_head_v1_state_capture.sbatch")
rover=$(submit --dependency="afterok:$smoke" --array=0-1 --mem=96G --time=12:00:00 \
  --job-name=vhv1cap_rover --export=ALL,DOMAIN=rover \
  "$REPO/scripts/value_head_v1_state_capture.sbatch")
mprime=$(submit --dependency="afterok:$smoke" --array=0-1 --mem=120G --time=12:00:00 \
  --job-name=vhv1cap_mprime --export=ALL,DOMAIN=mprime \
  "$REPO/scripts/value_head_v1_state_capture.sbatch")
finalize=$(submit --dependency="afterok:$drone:$fo:$rover:$mprime" \
  "$REPO/scripts/value_head_v1_state_finalize.sbatch")
postcapture=$(submit --dependency="afterok:$finalize" \
  "$REPO/scripts/value_head_v1_postcapture_controller.sbatch")

ledger="$ROOT/v1_resubmission_20260918.tsv"
printf 'role\tjob_id\n' > "$ledger"
printf 'smoke\t%s\ndrone\t%s\nfo_counters\t%s\nrover\t%s\nmprime\t%s\nstate_finalize\t%s\npostcapture_controller\t%s\n' \
  "$smoke" "$drone" "$fo" "$rover" "$mprime" "$finalize" "$postcapture" >> "$ledger"
printf 'V1_CHAIN|smoke=%s|drone=%s|fo=%s|rover=%s|mprime=%s|finalize=%s|postcapture=%s\n' \
  "$smoke" "$drone" "$fo" "$rover" "$mprime" "$finalize" "$postcapture"
