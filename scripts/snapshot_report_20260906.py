"""Build current comparison/provenance artifacts; preserve historical snapshots."""
import csv, re, math, itertools, statistics, html
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'experiment_tracking';P=T/'mcts_progressive_widening_cross_domain'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stat(x):
 n=len(x);m=statistics.mean(x);h={5:2.776445,10:2.262157}[n]*statistics.stdev(x)/math.sqrt(n)
 p=sum(abs(sum(a*b for a,b in zip(s,x)))>=abs(sum(x))-1e-9 for s in itertools.product((-1,1),repeat=n))/2**n
 return m,m-h,m+h,p
def table(rows,cols):
 return '| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+'\n'.join('| '+' | '.join(str(r.get(c,'' )).replace('|','/').replace('\n',' ') for c in cols)+' |' for r in rows)+'\n'
fresh=read(T/'job_refresh_20260906.csv')+read(T/'additional_results_20260906.csv');jobs={r['job_id']:r for r in fresh}
pol=read(T/'experiment_results.csv');seedrows=[]
for j in fresh:
 if 'rover' not in j['job_name']:continue
 src=re.search(r'_src(\d+)',j['log_path']).group(1)
 matches=[r for r in pol if r['experiment_id']=='MAIN-VAL' and r['stage']=='stage2' and r['task_type']=='policy_eval' and r['endpoint']=='validation_selected' and r['source_training_job_id']==src]
 assert len(matches)==1,(src,len(matches))
 p=matches[0]
 seedrows.append(dict(domain='rover',value_head=p['value_head'],seed=p['seed'],policy_score=p['score'],mcts_30m=j['success_30m'],mcts_2h=j['success_2h'],mcts_6h=j['success_6h'],job_id=j['job_id'],slurm_state=j['state'],val_valid=j['val_valid'],val_invalid=j['val_invalid'],policy_log=p['source_evaluation_log'],training_log=p['source_training_log'],checkpoint=p['checkpoint'],mcts_log=j['log_path'],completion_record=j['completion_record_path']))
write(T/'rover_validation_seed_results_20260906.csv',seedrows)
cmp=read(T/'stage2_policy_mcts_comparison_by_branch_20260905.csv')
for row in cmp:
 if row['domain']=='rover' and row['stage2_branch']=='validation_led':
  group=[r for r in seedrows if r['value_head']==row['value_head']];assert len(group)==10
  assert all(r['val_valid']==r['mcts_6h'] and r['val_invalid']=='0' for r in group)
  m,lo,hi,p=stat([float(r['mcts_6h'])-float(r['policy_score']) for r in group])
  row.update(n_terminal=10,policy_mean=statistics.mean(float(r['policy_score']) for r in group),mcts_30m=statistics.mean(float(r['mcts_30m']) for r in group),mcts_2h=statistics.mean(float(r['mcts_2h']) for r in group),mcts_6h=statistics.mean(float(r['mcts_6h']) for r in group),change_6h=m,ci95_low=lo,ci95_high=hi,raw_signflip_p=p,holm_p='',status='complete',conclusion='Positive mean; exact paired test determines evidence',row_level_provenance='experiment_tracking/rover_validation_seed_results_20260906.csv')
 if row['domain']=='counters' and row['stage2_branch']=='terminal_led':
  group=[r for r in fresh if 'SR10M_src204894' in r['job_name'] and ('novh' in r['job_name'])==(row['value_head']=='off')]
  row['n_terminal']=sum(r['state']!='RUNNING' for r in group)
  for k in ['30m','2h','6h']:row['mcts_'+k]='>='+str(round(sum(int(r['success_'+k]) for r in group)/10,2))
  row.update(status='live;one failed partial needs recovery' if row['value_head']=='on' else 'live',row_level_provenance='experiment_tracking/job_refresh_20260906.csv;experiment_tracking/experiment_results.csv')
 if row['domain']=='fo_counters' and row['stage2_branch']=='validation_led' and row['value_head']=='off':row['n_terminal']=7
write(T/'stage2_policy_mcts_comparison_by_branch_20260906.csv',cmp)
coverage=read(T/'stage2_mcts_branch_coverage_latest.csv')
for row in coverage:
 row['snapshot_time_idt']='2026-09-06T12:02:49+03:00'
 if row['domain']=='rover':row.update(validation_terminal=10,validation_live_submitted=0)
 if row['domain']=='fo_counters' and row['value_head']=='off':row.update(validation_terminal=7,validation_live_submitted=3)
 if row['domain']=='counters':
  row.update(terminal_terminal=4 if row['value_head']=='off' else 3,terminal_live_submitted=6 if row['value_head']=='off' else 7)
 row['row_level_provenance']+=';experiment_tracking/job_refresh_20260906.csv'
write(T/'stage2_mcts_branch_coverage_latest.csv',coverage);write(T/'stage2_mcts_branch_coverage_20260906.csv',coverage)
# The mixed-budget table is rebuilt from separately named fixed/PW20 columns.
mixed=read(P/'comparative_summary_20260901_1022.csv');follow=read(P/'pw70_followup_cutoffs_20260904.csv')
for row in follow:
 if row['job_id'] in jobs:
  j=jobs[row['job_id']];row['slurm_state']=j['state']
  for k,s in [('1800','30m'),('7200','2h'),('21600','6h')]:row['successes_le_'+k+'s']=j['success_'+s]
write(P/'pw70_followup_cutoffs_20260906.csv',follow)
for row in mixed:
 group=[r for r in follow if (r['domain'],r['stage'],r['value_head'])==(row['domain'],row['stage'],row['value_head'])]
 if group:
  live=any(r['slurm_state']=='RUNNING' for r in group)
  for k,s in [('1800','30m'),('7200','2h'),('21600','6h')]:row['pw70_'+s]=('>=' if live else '')+str(statistics.mean(float(r['successes_le_'+k+'s']) for r in group))
  row['status']='live' if live else 'terminal'
 row['notes']+='; fixed/PW20 column mix-up in 5 September report corrected; sources: terminal_results_20260901.csv and pw70_followup_cutoffs_20260906.csv'
write(P/'mixed_pw20_pw70_comparison_20260906.csv',mixed)
confirm=read(P/'pw70_live_summary_20260905.csv')
fixed=read(T/'mcts_counters_width_sensitivity/stage2_narrow_matched_10seed.csv')
man=read(P/'pw70_confirmatory_expansion_manifest.csv');screen=read(P/'terminal_results_20260901.csv');cs=[]
for vh in ('off','on'):
 for m in [r for r in man if r['domain']=='counters' and r['value_head']==vh]:
  j=next(j for j in fresh if m['manifest_id'] in j['log_path'])
  cs.append(dict(value_head=vh,seed=m['seed'],policy=float(m['policy_score']),pw30=float(j['success_30m']),pw2=float(j['success_2h']),pw6=float(j['success_6h']),job_id=j['job_id'],log=j['log_path'],completion=j['completion_record_path']))
 for s in [r for r in screen if r['domain']=='counters' and r['stage']=='stage2' and r['value_head']==vh]:
  f=next(r for r in follow if r['domain']=='counters' and r['stage']=='stage2' and r['value_head']==vh and r['seed']==s['seed'])
  cs.append(dict(value_head=vh,seed=s['seed'],policy=float(s['policy_score']),pw30=float(f['successes_le_1800s']),pw2=float(f['successes_le_7200s']),pw6=float(f['successes_le_21600s']),job_id=f['job_id'],log=f['source_log'],completion=''))
 group=[r for r in cs if r['value_head']==vh]
 target=next(r for r in confirm if r['domain']=='counters' and r['stage']=='stage2' and r['value_head']==vh)
 target['n_terminal']=5;target['status']='terminal conservative scores; OOM/timeout evidence retained'
 matched_fixed=[r for r in fixed if r['value_head']==vh and r['seed'] in {x['seed'] for x in group}]
 assert len(matched_fixed)==5
 for k in ('30m','2h','6h'):target['fixed_'+k]=statistics.mean(float(r['narrow_'+k]) for r in matched_fixed)
 for col,key in [('pw70_30m','pw30'),('pw70_2h','pw2'),('pw70_6h','pw6')]:target[col]=statistics.mean(r[key] for r in group)
 m,lo,hi,p=stat([r['pw6']-r['policy'] for r in group]);target.update(pw_vs_policy_6h=m,ci95_low=lo,ci95_high=hi,raw_p=p,row_level_provenance='experiment_tracking/mcts_progressive_widening_cross_domain/counters_pw70_five_seed_20260906.csv')
 target['pw_vs_fixed_6h']=target['pw70_6h']-target['fixed_6h']
 target['conclusion']='Matches fixed6h but below policy' if vh=='off' else 'Below fixed and policy means; no expansion now'
fixed_audit_path=T/'pw_fixed_five_seed_cutoff_audit_20260906.csv'
if fixed_audit_path.exists():
 audit=read(fixed_audit_path);base=read(T/'stage1_mcts_results.csv')
 for target in confirm:
  if target['domain'] not in {'fo_counters','rover'}:continue
  ids={r['mcts_job_id'] for r in base if r['domain']==target['domain'] and r['value_head']==target['value_head']}
  g=[r for r in audit if r['job_id'] in ids]
  assert len(g)==5
  for k in ('30m','2h','6h'):
   if all(r['completion_record_path'] for r in g):target['fixed_'+k]=statistics.mean(float(r['success_'+k]) for r in g)
   elif k!='6h':target['fixed_'+k]='timing unavailable'
  target['row_level_provenance']+=';experiment_tracking/pw_fixed_five_seed_cutoff_audit_20260906.csv'
write(P/'counters_pw70_five_seed_20260906.csv',cs)
write(P/'pw70_confirmation_summary_20260906.csv',[r for r in confirm if r['n_target']=='5'])
print('ROVER',[(r['value_head'],r['policy_mean'],r['mcts_6h'],r['raw_signflip_p']) for r in cmp if r['domain']=='rover' and r['stage2_branch']=='validation_led'])
print('COUNTERS PW',[(r['value_head'],r['pw70_30m'],r['pw70_2h'],r['pw70_6h']) for r in confirm if r['domain']=='counters' and r['stage']=='stage2'])
