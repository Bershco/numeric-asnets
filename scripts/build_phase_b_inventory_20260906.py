import csv
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'experiment_tracking'
B=R/'mprime_validation_ipc_scale_v1';rows=[]
for r in csv.DictReader((B/'validation_test_checkpoint_audit.csv').open(encoding='utf-8-sig')):
 rows.append(dict(lineage=f"stage1-{r['value_head']}-{r['seed']}",value_head=r['value_head'],seed=r['seed'],epoch=r['epoch'],checkpoint=r['checkpoint'],training_job=r['training_job'],training_log=r['training_log'],policy_job=r['evaluation_job'],policy_log=r['evaluation_log']))
for r in csv.DictReader((B/'validation_adequacy_phase_a_stage2_checkpoints_20260903.csv').open(encoding='utf-8-sig')):
 rows.append(dict(lineage=f"{r['branch']}-{r['value_head']}-{r['seed']}",value_head=r['value_head'],seed=r['seed'],epoch=r['epoch'],checkpoint=r['checkpoint'],training_job=r['training_job_id'],training_log=r['training_log'],policy_job=r['policy_job_id'],policy_log=r['policy_log']))
unique={(r['lineage'],r['checkpoint']):r for r in rows};rows=list(unique.values())
assert len(rows)==1130 and len({r['lineage'] for r in rows})==60,(len(rows),len({r['lineage'] for r in rows}))
with (R/'mprime_validation_phase_b_20260906/checkpoints.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('60 lineages /1130 checkpoints /2260 checkpoint-replicate evaluations')
