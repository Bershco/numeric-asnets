"""Materialize the five unused matched seeds for FO/Rover PW70."""
import csv
from pathlib import Path
from build_pw70_confirmatory_expansion import common, FIELDS
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'experiment_tracking'
used = {'1963100312','2011206605','534933607','923500475','1073581256'}
rows = []
for src in csv.DictReader((T/'experiment_results.csv').open(encoding='utf-8-sig')):
    if (src['experiment_id']=='MAIN-VAL' and src['task_type']=='policy_eval'
        and src['stage']=='stage1' and src['endpoint']=='validation_selected'
        and src['domain'] in {'fo_counters','rover'} and src['seed'] not in used):
        row=common(src,domain=src['domain'],stage='stage1')
        row.update(experiment_id='MCTS-PW70-TEN-SEED',
          manifest_id=f"pw70-ten-{src['domain']}-{src['value_head']}-{src['seed']}-s1",
          source_checkpoint=src['checkpoint'],source_training_job_id=src['source_training_job_id'],
          snapshot_epoch=src['epoch'],policy_score=src['score'],
          notes='Complete all ten original seeds; five new seeds per cell. Existing five retained. No outcome-based seed selection.')
        rows.append(row)
assert len(rows)==20 and len({r['manifest_id'] for r in rows})==20
p=T/'mcts_progressive_widening_cross_domain/pw70_ten_seed_expansion_20260906.csv'
with p.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
print(p, len(rows))
