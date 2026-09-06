"""Validation-only rescore of immutable checkpoint inventory on frozen A/B sets."""
import argparse,csv,hashlib,json,os,re,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 frozen=list(csv.DictReader((a.root/'frozen_validation_manifest.csv').open()))
 assert len(frozen)==60
 for r in frozen:assert hashlib.sha256((a.root/'candidates'/r['file']).read_bytes()).hexdigest()==r['sha256']
 inventory=list(csv.DictReader((a.root/'checkpoints.csv').open()))
 keys=sorted({r['lineage'] for r in inventory});assert len(keys)==60
 group=[r for r in inventory if r['lineage']==keys[a.task]]
 if a.preflight:group=group[:1]
 out=a.root/'rescore'/keys[a.task];out.mkdir(parents=True,exist_ok=True)
 for r in group:
  assert Path(r['checkpoint']).exists(),r['checkpoint']
  for rep in range(2):
   module=f'mprime_phase_b_20260906_{rep}'
   log=out/f"epoch_{r['epoch']}_rep{rep}.log";summary=log.with_suffix('.val.csv');done=log.with_suffix('.done.json')
   identity=dict(checkpoint=r['checkpoint'],replicate=rep,manifest_sha256=hashlib.sha256((a.root/'frozen_validation_manifest.csv').read_bytes()).hexdigest())
   if done.exists() and json.loads(done.read_text())==identity and summary.exists():continue
   complete=log.exists() and re.search(r'\[EVAL FINAL\].*?success=\d+(?:\.\d+)?/30(?:\.0)?',log.read_text(errors='replace'))
   if not complete:
    cmd=['./run_experiment','experiments_numeric.architecture_2.mprime',f'experiments_numeric.domain.{module}','--resume-from',r['checkpoint'],'--num-workers','3','--jpddl-max-heap','4g','--random-seed',r['seed'],'--worker-logs']
    if r['value_head']=='off':cmd+=['--disable-value-head']
    with log.open('w') as f:subprocess.run(cmd,cwd=a.repo/'asnets',stdout=f,stderr=subprocess.STDOUT,check=True)
   content=log.read_text(errors='replace')
   assert re.search(r'\[EVAL FINAL\].*?success=\d+(?:\.\d+)?/30(?:\.0)?',content),'missing full evaluation summary'
   assert not re.search(r'Worker (?:died without result|crashed)|CRASH_EXIT',content),'worker failure'
   subprocess.run(['python',str(a.repo/'asnets/tools/validate_eval_log_with_summary.py'),'--log',str(log),'--domain',module,'--validator','/home/hersco/tools/VAL/build/bin/Validate','--summary-csv',str(summary)],cwd=a.repo/'asnets',check=True)
   assert summary.exists() and summary.stat().st_size>0
   done.write_text(json.dumps(identity,sort_keys=True))
   print('VALIDATED',keys[a.task],r['epoch'],rep,flush=True)
if __name__=='__main__':main()
