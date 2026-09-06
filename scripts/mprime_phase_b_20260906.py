"""Frozen candidate generation and planner-only screening for two validation sets.

No checkpoint or test score is read by generation or screening. Goal distance is
a graph lower bound, not a claim about optimal plan length. ENHSP supplies a
solvability witness and VAL certifies it before an instance can be selected.
"""
import argparse, csv, hashlib, json, random, re, subprocess, sys
from pathlib import Path
from collections import deque

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def generate(root):
    rows=[]
    for rep in range(2):
      for tier,(nf,np,nv,distance) in enumerate(((10,10,2,2),(16,24,3,3),(22,40,4,4))):
       for idx in range(40):
        seed=906260000+rep*10000+tier*100+idx
        r=random.Random(seed); foods=[f'f{i}' for i in range(nf)]
        pains=[f'p{i}' for i in range(np)]; pleasures=[f'v{i}' for i in range(nv)]
        order=foods[:];r.shuffle(order)
        edges={(order[i],order[(i+1)%nf]) for i in range(nf)}
        # Sparse additional edges; every food remains reachable on the ring.
        for i in range(nf//5): edges.add(tuple(r.sample(foods,2)))
        pos={v:r.choice(foods) for v in pleasures}
        dist={x:999 for x in foods}; q=deque()
        for x in pos.values():dist[x]=0;q.append(x)
        while q:
            x=q.popleft()
            for a,b in sorted(edges):
                if a==x and dist[b]>dist[a]+1:dist[b]=dist[a]+1;q.append(b)
        targets=[x for x in foods if dist[x]>=distance]
        if not targets:continue
        cravings={(v,x) for v,x in pos.items()}
        for p in pains:cravings.add((p,r.choice(foods)))
        goals=[]
        for p in r.sample(pains,tier+1):
            eligible=[x for x in targets if (p,x) not in cravings]
            if eligible:goals.append((p,r.choice(eligible)))
        if len(goals)!=tier+1:continue
        init=[f'(= (locale {x}) {r.randint(1,5)})' for x in foods]
        init += [f'(= (harmony {v}) {r.randint(1,3)})' for v in pleasures]
        init += [f'(eats {a} {b})' for a,b in sorted(edges)]
        init += [f'(craves {a} {b})' for a,b in sorted(cravings)]
        name=f'mprime-b-{rep}-{tier}-{idx}'
        text=f'(define (problem {name}) (:domain mystery-prime-typed)\n(:objects '+ ' '.join(foods)+' - food '+ ' '.join(pleasures)+' - pleasure '+ ' '.join(pains)+' - pain)\n(:init '+' '.join(init)+')\n(:goal (and '+' '.join(f'(craves {a} {b})' for a,b in goals)+')))\n'
        path=root/'candidates'/f'{name}.pddl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding='utf-8')
        rows.append(dict(replicate=rep,tier=tier,index=idx,seed=seed,foods=nf,pains=np,pleasures=nv,edges=len(edges),goals=len(goals),min_goal_distance=min(dist[b] for a,b in goals),file=path.name,sha256=hashlib.sha256(text.encode()).hexdigest()))
    write_csv(root/'candidates.csv',rows)
    (root/'protocol.json').write_text(json.dumps(dict(version='20260906',replicates=2,per_tier=10,candidate_order='ascending seed, first ten certified per tier',planner='hmrp-ha-gbfs',timeout_seconds=60,minimum_plan_lengths=[4,6,8],selection_uses_network_scores=False),indent=2))
    print('Candidates:',len(rows))

def screen(root,repo,task):
    sys.path.insert(0,str(repo))
    from problem_generator.audit_generated_instances_enhsp import load_enhsp
    ENHSP,Status,configs=load_enhsp(repo)
    rows=list(csv.DictReader((root/'candidates.csv').open()))
    selected=[];out=[];rep,tier=divmod(task,3)
    domain=repo/'problems/numeric/mprime/domain.pddl'
    for row in rows:
        if (int(row['replicate']),int(row['tier']))!=(rep,tier):continue
        problem=root/'candidates'/row['file']
        assert hashlib.sha256(problem.read_bytes()).hexdigest()==row['sha256']
        planfile=root/'planner'/f"{problem.stem}.plan";planfile.parent.mkdir(exist_ok=True)
        try:
            result=ENHSP(configs['hmrp-ha-gbfs']+' -timeout 60').plan(str(domain),str(problem))
            plan=list(result.plan or [])
            planfile.write_text('\n'.join(f'{i}: ({str(a).strip().strip("()")} )' for i,a in enumerate(plan))+'\n')
            val=subprocess.run(['/home/hersco/tools/VAL/build/bin/Validate',str(domain),str(problem),str(planfile)],capture_output=True,text=True,timeout=30)
            vlog=planfile.with_suffix('.val.log');vlog.write_text(val.stdout+val.stderr)
            valid='Plan valid' in val.stdout and result.status==Status.SUCCESS
            row.update(planner_status=str(result.status),plan_length=len(plan),val_valid=valid,plan_file=str(planfile),val_log=str(vlog),job_id=__import__('os').environ.get('SLURM_JOB_ID','local'))
        except Exception as e:
            row.update(planner_status=repr(e),plan_length=0,val_valid=False,plan_file=str(planfile),val_log='',job_id=__import__('os').environ.get('SLURM_JOB_ID','local'))
        out.append(row);write_csv(root/f'planner_task_{task}.csv',out)
        if row['val_valid'] and row['plan_length'] >= [4,6,8][tier]:selected.append(row)
        if len(selected)==10:break
    if len(selected)!=10:raise RuntimeError(f'Only {len(selected)}/10 certified candidates in task {task}; do not freeze or score networks')
    write_csv(root/f'selected_task_{task}.csv',selected)
    print('CERTIFIED',task,len(selected),flush=True)

def freeze(root):
    rows=[]
    for task in range(6):
        group=list(csv.DictReader((root/f'selected_task_{task}.csv').open()))
        assert len(group)==10
        rows+=group
    assert len({r['sha256'] for r in rows})==60
    for r in rows:assert hashlib.sha256((root/'candidates'/r['file']).read_bytes()).hexdigest()==r['sha256']
    write_csv(root/'frozen_validation_manifest.csv',rows)
    print('FROZEN 60 independently generated, planner/VAL-certified validation instances')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','screen','freeze']);p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path);p.add_argument('--task',type=int,default=0);a=p.parse_args()
    if a.mode=='generate':generate(a.root)
    elif a.mode=='screen':screen(a.root,a.repo,a.task)
    else:freeze(a.root)
