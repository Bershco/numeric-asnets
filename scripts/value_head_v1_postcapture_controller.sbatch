#!/bin/bash
# Run only after the eight state-capture tasks and state finalizer succeed.
#SBATCH --partition=main
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/postcapture_%j.txt

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
EXCLUDE='ise-cpu-intl-[01,05,15,18,24,26]'
READY="$ROOT/ready/v1_checkpoint_candidates.ready.csv"
test -f "$READY"
READY_SHA=$(sha256sum "$READY" | awk '{print $1}')
mkdir -p "$ROOT/resource_preflight" "$ROOT/ready"
submit_preflight() {
  local domain=$1 mem=$2 hours=$3
  local raw
  raw=$(sbatch --parsable --array=0-1 --mem="$mem" --time="$hours" --exclude="$EXCLUDE" \
    --job-name="vhv1pre_${domain}" \
    --export="ALL,DOMAIN=$domain,READY_CANDIDATES=$READY,EXPECTED_CANDIDATES_SHA256=$READY_SHA" \
    "$REPO/scripts/value_head_v1_resource_preflight.sbatch")
  printf '%s\n' "${raw%%;*}"
}
DRONE=$(submit_preflight drone 48G 08:00:00)
FO=$(submit_preflight fo_counters 64G 08:00:00)
ROVER=$(submit_preflight rover 96G 12:00:00)
MPRIME=$(submit_preflight mprime 120G 12:00:00)
GATE="$ROOT/ready/v1_resource_gate.json"
FINAL=$(sbatch --parsable --dependency="afterok:$DRONE:$FO:$ROVER:$MPRIME" --exclude="$EXCLUDE" \
  --time=00:30:00 --mem=4G --cpus-per-task=1 \
  --job-name=vhv1pre_finalize \
  --wrap="python3 '$REPO/scripts/finalize_value_head_v1_resource_preflight.py' --preflight-root '$ROOT/resource_preflight' --candidates '$READY' --output '$GATE'")
FINAL=${FINAL%%;*}
RELEASE=$(sbatch --parsable --dependency="afterok:$FINAL" --exclude="$EXCLUDE" \
  --export="ALL,READY_CANDIDATES=$READY,RESOURCE_GATE=$GATE" \
  "$REPO/scripts/value_head_v1_release_controller.sbatch")
RELEASE=${RELEASE%%;*}
printf 'role\tjob_id\n' > "$ROOT/resource_submission_ledger.tsv"
printf 'drone\t%s\nfo_counters\t%s\nrover\t%s\nmprime\t%s\ngate\t%s\nrelease\t%s\n' \
  "$DRONE" "$FO" "$ROVER" "$MPRIME" "$FINAL" "$RELEASE" >> "$ROOT/resource_submission_ledger.tsv"
printf 'V1_PREFLIGHT_CHAIN|drone=%s|fo=%s|rover=%s|mprime=%s|gate=%s|release=%s\n' \
  "$DRONE" "$FO" "$ROVER" "$MPRIME" "$FINAL" "$RELEASE"
