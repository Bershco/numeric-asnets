#!/bin/bash
#SBATCH --job-name=SNAPSHOT_VAL_20260906
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-09-06/mprime_phase_b/posthoc_val_%j.log
set -euo pipefail
python3 - <<'PY'
import csv,subprocess
from pathlib import Path
root=Path('/home/hersco/training_new_domains/2026-09-06/mprime_phase_b')
repo=Path('/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context')
rows=list(csv.DictReader((root/'job_refresh_20260906.csv').open(encoding='utf-8-sig')))
for r in rows:
 if r['job_id'] not in {'20974363','20863187','20838267'}:continue
 subprocess.run(['python3',str(repo/'asnets/tools/validate_eval_log_with_summary.py'),'--log',r['log_path'],'--domain','counters','--validator','/home/hersco/tools/VAL/build/bin/Validate','--asnets-dir',str(repo/'asnets'),'--allow-incomplete','--summary-csv',str(root/f"val_{r['job_id']}.csv")],check=True)
PY
