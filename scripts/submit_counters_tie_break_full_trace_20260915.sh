#!/usr/bin/env bash
# Submit only after the current code commit is deployed to the isolated checkout.
set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break
actual_commit=$(git -C "$repo" rev-parse HEAD)
[[ "$actual_commit" == "$CODE_COMMIT" ]]
sbatch --parsable \
  --dependency=afterany:21233925:21308619 \
  --export="ALL,CODE_COMMIT=$CODE_COMMIT" \
  "$repo/scripts/counters_tie_break_full_trace_controller_20260915.sbatch"
