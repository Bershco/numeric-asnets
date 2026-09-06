"""Publish the snapshot with explicit source links and all registered experiments."""
import csv,html,re,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'experiment_tracking';P=T/'mcts_progressive_widening_cross_domain'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(rows,cols):
 return '| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+'\n'.join('| '+' | '.join(str(r.get(c,'' )).replace('|','/').replace('\n',' ') for c in cols)+' |' for r in rows)+'\n'
def fmt(x):
 try:return f'{float(x):.3g}'
 except:return str(x)
def link(p):return f'[{p.name}]({p.resolve().as_posix()})'
old={r['job_id']:r for r in read(T/'cluster_workload_latest.csv')};queue=[]
qlines=(T/'queue_only_20260906.txt').read_text(encoding='utf-8-sig').splitlines()
stamp=next((x for x in qlines if x.startswith('2026-') and '|' not in x),'2026-09-06T12:21:32+03:00')
for line in qlines:
 bits=line.split('|')
 if len(bits)!=8:continue
 jid,name,state,cpu,mem,elapsed,limit,node=bits
 exp=old.get(jid,{}).get('experiment','')
 if name.startswith('pw70-ten-'):exp='PW70 ten-seed FO/Rover expansion'
 if name.startswith('MPRIME'):exp='MPrime Phase B validation'
 if not exp:exp=name
 log=old.get(jid,{}).get('log_path','')
 if name.startswith('pw70-ten-'):log=f'/home/hersco/training_new_domains/2026-08-30/mcts_pw_cross_domain/{jid}_{name}.txt'
 if name.startswith('MPRIME'):log='/home/hersco/training_new_domains/2026-09-06/mprime_phase_b/'
 queue.append(dict(snapshot_time_idt=stamp,experiment=exp,job_id=jid,job_name=name,state=state,cpus=cpu,memory_gib=float(mem.rstrip('G')),elapsed=elapsed,time_limit=limit,node_or_reason=node,log_path=log))
write(T/'cluster_workload_latest.csv',queue);write(T/'cluster_workload_20260906.csv',queue)
summary=[]
for state in ('RUNNING','PENDING'):
 group=[r for r in queue if r['state']==state]
 summary.append(dict(state=state,jobs=len(group),cpus=sum(int(r['cpus']) for r in group),memory_gib=sum(r['memory_gib'] for r in group),provenance='cluster_workload_20260906.csv'))
write(T/'cluster_workload_summary_latest.csv',summary)
groups=[]
for exp in sorted({r['experiment'] for r in queue}):
 group=[r for r in queue if r['experiment']==exp]
 groups.append(dict(experiment=exp,running=sum(r['state']=='RUNNING' for r in group),pending=sum(r['state']=='PENDING' for r in group),requested_cpus=sum(int(r['cpus']) for r in group),requested_gib=sum(r['memory_gib'] for r in group),provenance='cluster_workload_20260906.csv'))
write(T/'live_experiment_status_latest.csv',groups)
registry=read(T/'experiment_registry.csv')
for row in registry:
 if row['experiment_id']=='MCTS-PW70-CONFIRMATORY':row.update(status='completed-five-seed-screen',next_action='All five seed cells terminal; expand FO/Rover to ten with separately registered20 jobs; Counters PW70 does not match policy consistently.')
 if row['experiment_id']=='MCTS-STAGE2-BRANCH-COMPLETION':row.update(status='40-terminal-17-running',scope='All57 identities submitted; BG13 terminal Rover19 terminal FO1 new terminal; Counters7 terminal including1 failed partial',next_action='FO4 and Counters13 run; include original FO terminal2 separately. Failed Counters20974363 has22 VAL-valid partial successes and needs infrastructure-aware remaining-instance recovery.')
 if row['experiment_id']=='MPRIME-VAL-ADEQUACY':row.update(status='phase-b-planner-screen-live',manifest_path='experiment_tracking/mprime_validation_phase_b_20260906/checkpoints.csv',held_reason='',next_action='240 candidates created; planner array21039224; afterok finalizer21039342 freezes60 instances and schedules preflight then60 rescore lineages; no MPrime MCTS authorized.')
 if row['experiment_id']=='MCTS-PW-COUNTERS-DIVERGENCE':row.update(status='pw20-complete-pw70-one-terminal-two-live',next_action='PW70 seed923500475 TIMEOUT21/59; other2 live. One requeued at06:03; preserve total budget/restart provenance.')
 if row['experiment_id']=='MCTS-PW-30M':row.update(status='held-optional-execution-validation',next_action='Post-hoc cutoffs suffice for solutions-within-budget coverage; fresh runs needed only for whole-job resource/time validation or missing/censored timing evidence.')
new=dict(registry[0]);new.update(experiment_id='MCTS-PW70-TEN-SEED',display_name='PW70 ten-seed FO Counters/Rover extension',role='search-sensitivity',status='submitted',scope='FO Counters and Rover S1,2 VH,5 new seeds per cell;20 jobs',primary_question='Do five-seed PW70 findings survive all ten original seeds?',configuration_summary='Kmin3 c0.6 alpha0.5 widthmax20 70sim SAFE1;3workers120GiB6h',results_file='experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_submissions_20260906.tsv',manifest_path='experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_expansion_20260906.csv',held_reason='',next_action='Jobs21039201-21039220; reuse the existing five seeds; do not select subsets by outcomes.')
registry=[r for r in registry if r['experiment_id']!=new['experiment_id']]+[new]
write(T/'experiment_registry.csv',registry);write(T/'experiments.csv',registry)
coverage_plan=read(T/'evaluation_coverage_plan.csv')
for row in coverage_plan:
 if row['experiment_id']=='MAIN-VAL':row.update(status='Policy complete: all100 Stage2 lineages; Rover n10 restored. MCTS BG/Drone/Rover complete; FO live; Counters complete.',next_action='Finish four FO validation MCTS jobs; retain all original/retry log pointers.')
 if row['experiment_id']=='MAIN-TERM':row.update(status='Policy complete; MCTS BG/Drone/Rover complete; FO2 and Counters13 live.',next_action='Finish MCTS tails; one Counters partial failed and requires budget-aware recovery.')
write(T/'evaluation_coverage_plan.csv',coverage_plan)
cmp=read(T/'stage2_policy_mcts_comparison_by_branch_20260906.csv');display=[]
for r in cmp:
 if r['domain']=='fo_counters' and r['value_head']=='off' and r['stage2_branch']=='validation_led':r['n_terminal']='7'
 live='live' in r['status']
 scores=' / '.join(('≥' if live and not str(r[k]).startswith('>=') else '')+fmt(r[k]).replace('>=','≥') for k in ('mcts_30m','mcts_2h','mcts_6h'))
 display.append({'Cell':r['domain']+'/'+r['value_head'],'Branch':r['stage2_branch'],'Search':r['search'],'n terminal':r['n_terminal'],'Policy':fmt(r['policy_mean']),'MCTS30m/2h/6h':scores,'Δ [95% CI]':f"{fmt(r['change_6h'])} [{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]" if r['change_6h'] else 'live','raw p':fmt(r['raw_signflip_p']),'Conclusion':r['conclusion']})
sections=['# Complete experiment snapshot — '+stamp+'\n', '## Current workload\n'+table(summary,['state','jobs','cpus','memory_gib']),table(groups,['experiment','running','pending','requested_cpus','requested_gib']),
'## Actions and integrity findings\n20 approved PW70 jobs submitted21039201–21039220; each6CPUs120GiB72h. MPrime planner array21039224 has six2CPU8GiB90minute tasks. Finalizer21039342 is afterok-dependent and will freeze sets, run a two-replicate checkpoint preflight, then release60 validation-only lineage jobs with at most12 concurrent.\n\nThree partial jobs20974363,20863187,20838267 passed post-hoc VAL in job21039344:22,21,29 unique valid successes, zero invalid. No inference repeated. Counters20974363 failed after OOM/native−4 on requeue; its partial record is preserved. Two jobs20863185 and20974364 were requeued at06:03; their reset elapsed times are not total campaign compute.\n\nAll19 Rover evaluations are terminal and have matching VAL summaries including OOM-labelled allocations. These are fixed-budget results; interrupted work is never silently counted as successful.\n',
'## Stage-2 MCTS — every domain and both branches\n'+table(display,list(display[0])),
'## PW20 and PW70 — identical two-seed cells\n'+table(read(P/'mixed_pw20_pw70_comparison_20260906.csv'),['domain','stage','value_head','n','policy_mean','fixed_search','fixed_30m','fixed_2h','fixed_6h','pw20_30m','pw20_2h','pw20_6h','pw70_30m','pw70_2h','pw70_6h','status']),
'Correction: the5 September report used PW20 values in several fixed-comparator columns. The underlying distinct columns are preserved and used here. No interpretation should rely on the swapped columns.\n',
'## PW70 five-seed conclusions\n'+table(read(P/'pw70_confirmation_summary_20260906.csv'),['domain','stage','value_head','policy_mean','fixed_30m','fixed_2h','fixed_6h','pw70_30m','pw70_2h','pw70_6h','pw_vs_policy_6h','ci95_low','ci95_high','raw_p']),
'FO/Rover are being expanded to10; Counters is not expanded. Five was a resource-saving screen and could not reach two-sided exactp<.05. Ten does not guarantee significance. Post-hoc30m/2h is defensible for observed time-to-solution coverage; it does not estimate fresh whole-job runtime or remove OOM/censoring.\n',
'## MPrime validation Phase B\n240 independent candidates across two pools and three structural tiers generated with frozen seeds/hashes. Goal locations are separated from all initial pleasure positions by graph distance; direct two-action witnesses are not inserted. ENHSP and VAL screen solvability before any network score is read. Each pool selects the first10 certified candidates per tier in seed order. This is a deliberately harder validation hypothesis, not proof of adequacy.\n\nThe complete inventory contains1130 saved checkpoints across60 lineages. Scoring both independent sets means2260 checkpoint-replicate evaluations, not1130. PhaseB uses existing weights. The60-lineage campaign assesses checkpoint selection; a separate follow-up rescore of the28 anchor candidates is needed to test anchor-rank stability. No MPrime MCTS submitted.\n',
'## Every registered experiment, including held and historical work\n'+table(registry,['experiment_id','status','scope','next_action','results_file'])]
oldmd=(T/'status_20260905_latest.md').read_text(encoding='utf-8')
completed=oldmd[oldmd.index('## Research questions'):oldmd.index('## Held designs')]
start=completed.index('### RQ2/RQ4')
end=completed.index('## Completed experiments',start)
completed=completed[:start]+'### RQ2/RQ4 — Stage-2 inference-time MCTS\nThe updated all-domain, both-branch table above supplies the current evidence. Rover validation is now complete. FO validation/on-terminal and Counters terminal remain live.\n\n'+completed[end:]
start=completed.index('These policy results are computationally complete')
end=completed.index('### Other completed side experiments',start)
completed=completed[:start]+'These policy results remain selector-provisional; Phase B is now running as described above.\n\n'+completed[end:]
sections.append('## Completed RQs and experiments — static results retained\n'+completed)
sections.append('## Long Drone detailed endpoints\n'+table(read(T/'long_drone_endpoint_results.csv'),['value_head','seed','checkpoint_role','cumulative_epoch','policy_score','mcts_score','notes']))
sections.append('## Provenance\nEvery current live row references its SlurmID and log. New comparisons reference row-level companions containing checkpoint, training log, policy log and MCTS log. '+link(T/'rover_validation_seed_results_20260906.csv')+'; '+link(T/'job_refresh_20260906.csv')+'; '+link(T/'snapshot_provenance_index_20260905.csv')+'. Older dated records remain unchanged.\n')
md='\n\n'.join(sections);(T/'status_20260906_latest.md').write_text(md,encoding='utf-8')
# Standalone readable tables with sticky headers and horizontal scrolling.
parts=[];lines=md.splitlines();i=0
while i<len(lines):
 line=lines[i]
 if line.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].startswith('|'):block.append(lines[i]);i+=1
  cells=lambda s:[html.escape(x.strip()) for x in s.strip('|').split('|')]
  parts.append('<div class="table"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in cells(block[0]))+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in cells(s))+'</tr>' for s in block[2:])+'</tbody></table></div>');continue
 if line.startswith('#'):
  n=len(line)-len(line.lstrip('#'));parts.append(f'<h{n}>'+html.escape(line[n:].strip())+f'</h{n}>')
 elif line.strip():parts.append('<p>'+html.escape(line)+'</p>')
 i+=1
page='<!doctype html><meta charset="utf-8"><title>Thesis experiments · 6 September 2026</title><style>body{font:16px system-ui;margin:32px;color:#192431;background:#fafbfd}h1,h2{color:#153e65}p{max-width:1100px;line-height:1.6}.table{overflow-x:auto;margin:20px 0;border:1px solid #ccd7e0}table{border-collapse:collapse;background:white;width:100%}th{background:#173e65;color:white;position:sticky;top:0}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid #dce3ea;min-width:80px}tr:nth-child(even){background:#eef4fa}</style>'+''.join(parts)
(T/'status_20260906.html').write_text(page,encoding='utf-8')
index=read(T/'snapshot_provenance_index_20260905.csv')
index.append(dict(aggregate_file='experiment_tracking/stage2_policy_mcts_comparison_by_branch_20260906.csv',scope='current Stage2 both branches',row_level_companion='experiment_tracking/rover_validation_seed_results_20260906.csv;experiment_tracking/job_refresh_20260906.csv;experiment_tracking/stage2_mcts_historical_log_audit_20260902.csv',backtrace_fields='job_id;checkpoint;training_log;policy_log;mcts_log;completion_record',status='complete'))
index.append(dict(aggregate_file='experiment_tracking/mcts_progressive_widening_cross_domain/mixed_pw20_pw70_comparison_20260906.csv',scope='PW20 versus PW70 and fixed',row_level_companion='experiment_tracking/mcts_progressive_widening_cross_domain/terminal_results_20260901.csv;experiment_tracking/mcts_progressive_widening_cross_domain/pw70_followup_cutoffs_20260906.csv',backtrace_fields='pw_job;fixed_job;source_pw_log;source_log',status='complete; lower-bound timing explicitly marked'))
write(T/'snapshot_provenance_index_20260906.csv',index)
print(summary)
