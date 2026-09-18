#!/usr/bin/env python3
import csv, itertools, json, math, pathlib, statistics, sys

root=pathlib.Path(sys.argv[1])
manifest=list(csv.DictReader((root/'manifest.csv').open()))
rows=[]
for arm in manifest:
    d=root/'outputs'/f"arm_{arm['arm_index']}_{arm['domain']}_{arm['seed']}_{arm['semantics']}"
    for p in sorted(d.glob('policy_epoch_*/result.json')):
        rows.append(json.loads(p.read_text()))
assert len(rows)==168, len(rows)
fields=['arm_index','domain','seed','semantics','epoch','score','total','source_checkpoint','source_checkpoint_sha256','checkpoint','evaluation_log']
with (root/'learning_curve_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(sorted(rows,key=lambda r:(r['domain'],r['seed'],r['semantics'],r['epoch'])))
end=[r for r in rows if r['epoch']==99]
with (root/'endpoint99_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(sorted(end,key=lambda r:(r['domain'],r['seed'],r['semantics'])))

def exact_signflip(ds):
    obs=abs(statistics.mean(ds)); vals=[]
    for signs in itertools.product((-1,1),repeat=len(ds)):
        vals.append(abs(statistics.mean([d*s for d,s in zip(ds,signs)])))
    return sum(v>=obs-1e-12 for v in vals)/len(vals)

summary=[]
for domain in sorted({r['domain'] for r in end}):
    by={(r['seed'],r['semantics']):r for r in end if r['domain']==domain}
    seeds=sorted({k[0] for k in by})
    ds=[by[(s,'deterministic_current')]['score']-by[(s,'legacy_dropout_current')]['score'] for s in seeds]
    mean=statistics.mean(ds); sd=statistics.stdev(ds); half=12.706*sd/math.sqrt(2)
    summary.append(dict(domain=domain,n=2,legacy_mean=statistics.mean(by[(s,'legacy_dropout_current')]['score'] for s in seeds),deterministic_mean=statistics.mean(by[(s,'deterministic_current')]['score'] for s in seeds),det_minus_legacy=mean,ci95_low=mean-half,ci95_high=mean+half,exact_signflip_p=exact_signflip(ds),interpretation='two-seed pilot; no population claim'))
with (root/'domain_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
print(json.dumps(dict(curve_rows=len(rows),endpoint_rows=len(end),domains=summary),indent=2))
