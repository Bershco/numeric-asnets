#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/release_%j.txt

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${READY_CANDIDATES:?missing READY_CANDIDATES}"
: "${RESOURCE_GATE:?missing RESOURCE_GATE}"
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
read -r STATUS EXPECTED_SHA < <(python3 - "$RESOURCE_GATE" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
print(gate.get('status', ''), gate.get('candidate_manifest_sha256', ''))
PY
)
test "$STATUS" = pass
test "$(sha256sum "$READY_CANDIDATES" | awk '{print $1}')" = "$EXPECTED_SHA"
mkdir -p "$ROOT/science" "$ROOT/results"
submit_domain() {
  local domain=$1 mem=$2 hours=$3
  local raw
  raw=$(sbatch --parsable --array=0-3 --mem="$mem" --time="$hours" \
    --job-name="vhv1_${domain}" \
    --export="ALL,DOMAIN=$domain,READY_CANDIDATES=$READY_CANDIDATES,RESOURCE_GATE=$RESOURCE_GATE" \
    "$REPO/scripts/value_head_v1_science.sbatch")
  printf '%s\n' "${raw%%;*}"
}
DRONE=$(submit_domain drone 48G 08:00:00)
FO=$(submit_domain fo_counters 64G 08:00:00)
ROVER=$(submit_domain rover 96G 12:00:00)
MPRIME=$(submit_domain mprime 120G 12:00:00)
FINAL=$(sbatch --parsable --dependency="afterok:$DRONE:$FO:$ROVER:$MPRIME" \
  --export="ALL,READY_CANDIDATES=$READY_CANDIDATES,RESOURCE_GATE=$RESOURCE_GATE" \
  "$REPO/scripts/value_head_v1_results_finalize.sbatch")
FINAL=${FINAL%%;*}
printf 'domain\tjob_id\n' > "$ROOT/science_submission_ledger.tsv"
printf 'drone\t%s\nfo_counters\t%s\nrover\t%s\nmprime\t%s\nfinalizer\t%s\n' \
  "$DRONE" "$FO" "$ROVER" "$MPRIME" "$FINAL" >> "$ROOT/science_submission_ledger.tsv"
printf 'V1_SCIENCE_SUBMITTED|drone=%s|fo=%s|rover=%s|mprime=%s|finalizer=%s\n' \
  "$DRONE" "$FO" "$ROVER" "$MPRIME" "$FINAL"
