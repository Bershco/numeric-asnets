#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=MPRIME_S2_RECON
#SBATCH --output=/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_search/reconciliation_20260917/slurm/controller_%j.out

set -euo pipefail

: "${CODE_COMMIT:?CODE_COMMIT is required}"

CHECKOUT=${CHECKOUT:-/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context}
CAMPAIGN=${CAMPAIGN:-/home/hersco/training_new_domains/2026-09-16/mprime_final_stage2_search}
MANIFEST=${MANIFEST:-$CAMPAIGN/manifest.csv}
RECON_ROOT=${RECON_ROOT:-$CAMPAIGN/reconciliation_20260917}
MODE=${MODE:-initial}
EXCLUDE_NODES=${EXCLUDE_NODES:-ise-cpu128-03,ise-cpu-intl-[01,09-15,27]}

mkdir -p "$RECON_ROOT/slurm"
ACTUAL_COMMIT=$(git -C "$CHECKOUT" rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$CODE_COMMIT" ]]; then
  echo "Checkout commit $ACTUAL_COMMIT != declared $CODE_COMMIT" >&2
  exit 2
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$RECON_ROOT/$STAMP"
mkdir -p "$OUT"

set +e
python "$CHECKOUT/scripts/reconcile_mprime_final_stage2_search_20260917.py" \
  --manifest "$MANIFEST" --campaign-root "$CAMPAIGN" \
  --reconciliation-output "$OUT/reconciliation.csv" \
  --recovery-manifest "$OUT/recovery_manifest.csv" \
  --standard-recovery-manifest "$OUT/recovery_standard.csv" \
  --oom-recovery-manifest "$OUT/recovery_oom.csv" \
  --conflicts-output "$OUT/conflicts.csv" \
  --summary-json "$OUT/summary.json"
RECON_RC=$?
set -e

if [[ ! -f "$OUT/summary.json" ]]; then
  echo "Reconciliation failed before producing a summary (rc=$RECON_RC)." >&2
  exit "$RECON_RC"
fi

cp "$OUT/reconciliation.csv" "$RECON_ROOT/reconciliation_latest.csv"
cp "$OUT/recovery_manifest.csv" "$RECON_ROOT/recovery_manifest_latest.csv"
cp "$OUT/recovery_standard.csv" "$RECON_ROOT/recovery_standard_latest.csv"
cp "$OUT/recovery_oom.csv" "$RECON_ROOT/recovery_oom_latest.csv"
cp "$OUT/conflicts.csv" "$RECON_ROOT/conflicts_latest.csv"
cp "$OUT/summary.json" "$RECON_ROOT/summary_latest.json"

CONFLICTS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["conflicts"])' "$OUT/summary.json")
MISSING=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["unclassified"])' "$OUT/summary.json")
STANDARD_MISSING=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["standard_recoveries"])' "$OUT/summary.json")
OOM_MISSING=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["oom_recoveries"])' "$OUT/summary.json")
if (( CONFLICTS > 0 )); then
  echo "Refusing automatic recovery: $CONFLICTS conflicting terminal identities require manual review." >&2
  exit 3
fi
if (( RECON_RC != 0 )); then
  echo "Reconciliation failed with rc=$RECON_RC." >&2
  exit "$RECON_RC"
fi
if (( MISSING == 0 )); then
  touch "$RECON_ROOT/COMPLETE"
  echo "All 800 seed-instance identities are terminal; no recovery submitted."
  exit 0
fi
if [[ "$MODE" == "finalize" ]]; then
  echo "Exact recovery left $MISSING genuinely unclassified identities; see $OUT." >&2
  exit 4
fi
if [[ "$MODE" != "initial" ]]; then
  echo "Unknown MODE=$MODE" >&2
  exit 2
fi

if [[ ! -f "$RECON_ROOT/submissions.tsv" ]]; then
  echo -e "timestamp_utc\trole\tjob_id\tdependency_or_count\tartifact" > "$RECON_ROOT/submissions.tsv"
fi

RECOVERY_DEPENDENCY=""
if (( STANDARD_MISSING > 0 )); then
  STANDARD_JOB=$(sbatch --parsable \
    --array="0-$((STANDARD_MISSING - 1))" \
    --exclude="$EXCLUDE_NODES" \
    --export="ALL,CODE_COMMIT=$CODE_COMMIT,CHECKOUT=$CHECKOUT,CAMPAIGN=$CAMPAIGN,MANIFEST=$MANIFEST,RECOVERY_MANIFEST=$OUT/recovery_standard.csv" \
    "$CHECKOUT/scripts/mprime_final_stage2_search_exact_recovery_20260917.sbatch")
  RECOVERY_DEPENDENCY="$STANDARD_JOB"
  echo -e "$STAMP\tstandard_recovery\t$STANDARD_JOB\t$STANDARD_MISSING\t$OUT/recovery_standard.csv" >> "$RECON_ROOT/submissions.tsv"
fi
if (( OOM_MISSING > 0 )); then
  OOM_JOB=$(sbatch --parsable \
    --array="0-$((OOM_MISSING - 1))" \
    --mem=200G \
    --exclude="$EXCLUDE_NODES" \
    --export="ALL,CODE_COMMIT=$CODE_COMMIT,CHECKOUT=$CHECKOUT,CAMPAIGN=$CAMPAIGN,MANIFEST=$MANIFEST,RECOVERY_MANIFEST=$OUT/recovery_oom.csv" \
    "$CHECKOUT/scripts/mprime_final_stage2_search_exact_recovery_20260917.sbatch")
  if [[ -n "$RECOVERY_DEPENDENCY" ]]; then
    RECOVERY_DEPENDENCY="$RECOVERY_DEPENDENCY:$OOM_JOB"
  else
    RECOVERY_DEPENDENCY="$OOM_JOB"
  fi
  echo -e "$STAMP\toom_recovery\t$OOM_JOB\t$OOM_MISSING\t$OUT/recovery_oom.csv" >> "$RECON_ROOT/submissions.tsv"
fi

if [[ -z "$RECOVERY_DEPENDENCY" ]]; then
  echo "Internal error: $MISSING missing identities but no recovery array was materialized." >&2
  exit 5
fi

FINAL_JOB=$(sbatch --parsable \
  --dependency="afterany:$RECOVERY_DEPENDENCY" \
  --export="ALL,CODE_COMMIT=$CODE_COMMIT,CHECKOUT=$CHECKOUT,CAMPAIGN=$CAMPAIGN,MANIFEST=$MANIFEST,RECON_ROOT=$RECON_ROOT,MODE=finalize" \
  "$CHECKOUT/scripts/mprime_final_stage2_search_reconcile_controller_20260917.sbatch")
echo -e "$STAMP\tfinal_reconcile\t$FINAL_JOB\t$RECOVERY_DEPENDENCY\t$OUT" >> "$RECON_ROOT/submissions.tsv"
echo "Submitted exact recoveries (standard=$STANDARD_MISSING; OOM=$OOM_MISSING; jobs=$RECOVERY_DEPENDENCY) and final reconciliation $FINAL_JOB."
