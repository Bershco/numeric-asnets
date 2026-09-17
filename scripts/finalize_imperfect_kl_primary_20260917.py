#!/usr/bin/env python3
"""Join starting scores, new endpoints, and the two exact reused pairs."""
import csv,itertools,json,math,pathlib,statistics,sys
root,manifest=map(pathlib.Path,sys.argv[1:3])
rows=list(csv.DictReader(manifest.open(newline=""))); results=[]
reused={
 ('counters','534933607'):(6,35),
 ('counters','2082152039'):(59,38),
}
for row in rows:
    key=(row['domain'],row['seed'])
    if key in reused: legacy,det=reused[key]
    else:
        stem=f'lineage_{row["lineage_index"]}_{row["domain"]}_{row["seed"]}'
        legacy=json.loads((root/'outputs'/f'{stem}_legacy_capture'/'endpoint_result.json').read_text())['score']
        det=json.loads((root/'outputs'/f'{stem}_deterministic_current'/'endpoint_result.json').read_text())['score']
    results.append(dict(lineage_index=row['lineage_index'],domain=row['domain'],seed=row['seed'],total=row['total'],starting_policy=int(row['starting_policy_score']),legacy_epoch0=legacy,deterministic_epoch0=det,det_minus_legacy=det-legacy,reuse_status=row['reuse_status'],source_checkpoint=row['source_checkpoint'],source_checkpoint_sha256=row['source_checkpoint_sha256'],source_policy_job_id=row['source_policy_job_id']))
fields=list(results[0]); out=root/'primary_lineage_results.csv'
with out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
summary=[]
for domain in sorted({r['domain'] for r in results}):
    part=[r for r in results if r['domain']==domain]
    contrasts=(
        ('legacy_minus_start','starting_policy','legacy_epoch0'),
        ('deterministic_minus_start','starting_policy','deterministic_epoch0'),
        ('deterministic_minus_legacy','legacy_epoch0','deterministic_epoch0'),
    )
    for label,before,after in contrasts:
        delta=[r[after]-r[before] for r in part]; mean=statistics.fmean(delta); se=statistics.stdev(delta)/math.sqrt(len(delta)); critical=2.262
        observed=abs(mean); perm=[abs(statistics.fmean(sign*x for sign,x in zip(signs,delta))) for signs in itertools.product((-1,1),repeat=len(delta))]
        p=sum(x>=observed-1e-12 for x in perm)/len(perm)
        summary.append(dict(domain=domain,contrast=label,n=len(part),baseline_mean=statistics.fmean(r[before] for r in part),treatment_mean=statistics.fmean(r[after] for r in part),mean_paired_change=mean,ci95_low=mean-critical*se,ci95_high=mean+critical*se,exact_sign_flip_p=p,holm_p='',positive=sum(x>0 for x in delta),zero=sum(x==0 for x in delta),negative=sum(x<0 for x in delta)))
for contrast in sorted({r['contrast'] for r in summary}):
    family=[i for i,r in enumerate(summary) if r['contrast']==contrast]
    ordered=sorted(family,key=lambda i:summary[i]['exact_sign_flip_p']); running=0.0
    for rank,index in enumerate(ordered):
        adjusted=min(1.0,(len(ordered)-rank)*summary[index]['exact_sign_flip_p']); running=max(running,adjusted); summary[index]['holm_p']=running
with (root/'domain_summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
print(json.dumps({'status':'complete','lineages':len(results),'results':str(out),'summary':str(root/'domain_summary.csv')}))
