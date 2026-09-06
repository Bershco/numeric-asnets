"""Freeze planner-only sets, create new domain modules, gate rescoring on preflight."""
import csv,json,subprocess
from pathlib import Path
from mprime_phase_b_20260906 import freeze
ROOT=Path('/home/hersco/training_new_domains/2026-09-06/mprime_phase_b')
REPO=Path('/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context')
freeze(ROOT)
rows=list(csv.DictReader((ROOT/'frozen_validation_manifest.csv').open()))
for rep in range(2):
 paths=[str(ROOT/'candidates'/r['file']) for r in rows if int(r['replicate'])==rep]
 module=REPO/f'asnets/experiments_numeric/domain/mprime_phase_b_20260906_{rep}.py'
 content='from experiments_numeric.domain.mprime import *\nTEST_RUNS = '+repr([([p],None) for p in paths])+'\nVALIDATION_PDDLS = '+repr({'phase_b':paths})+'\n'
 if module.exists():assert module.read_text()==content
 else:module.write_text(content)
ledger=ROOT/'rescore_submission.json'
if ledger.exists():raise SystemExit('Already submitted: '+ledger.read_text())
batch=str(ROOT/'mprime_phase_b_rescore_20260906.sbatch')
pre=subprocess.check_output(['sbatch','--parsable','--export=ALL,PREFLIGHT=1','--time=02:00:00',batch],text=True).strip().split(';')[0]
ledger.write_text(json.dumps(dict(preflight=pre)))
full=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{pre}','--array=0-59%12','--export=ALL,PREFLIGHT=0',batch],text=True).strip().split(';')[0]
ledger.write_text(json.dumps(dict(preflight=pre,rescore=full)))
print(ledger.read_text())
