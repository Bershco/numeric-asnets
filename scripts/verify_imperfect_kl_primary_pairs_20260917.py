#!/usr/bin/env python3
"""Fail closed unless every new primary KL pair is exactly matched."""
import csv, hashlib, json, math, pathlib, sys

root, manifest, output = map(pathlib.Path, sys.argv[1:4])
def steps(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("record_type")=="optimizer_step"]
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def bits(step): return [r["target_pred_argmax_disagree_pre"] for b in step["batches"] for r in b["rows"]]
verified=[]
for row in csv.DictReader(manifest.open(newline="")):
    stem=f'lineage_{row["lineage_index"]}_{row["domain"]}_{row["seed"]}'
    left=steps(root/"outputs"/f"{stem}_legacy_capture"/"first_update_audit.jsonl")
    right=steps(root/"outputs"/f"{stem}_deterministic_current"/"first_update_audit.jsonl")
    assert len(left)==len(right)==60
    for i,(a,b) in enumerate(zip(left,right)):
        assert a["kl_current_forward"]=="training" and b["kl_current_forward"]=="deterministic"
        assert a["fixed_step_rng_step_seed"]==b["fixed_step_rng_step_seed"]
        assert sha(pathlib.Path(a["frozen_batch_file"]))==b["source_frozen_batch_sha256"]
    assert math.isclose(left[0]["policy_gradient_l2"],right[0]["policy_gradient_l2"],rel_tol=1e-6,abs_tol=1e-8)
    assert bits(left[0])==bits(right[0])
    verified.append({"lineage_index":row["lineage_index"],"domain":row["domain"],"seed":row["seed"],"steps":60,"ordered_batches_equal":True,"step0_policy_gradient_equal":True,"step0_targets_equal":True,"status":"verified"})
with output.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=verified[0].keys()); w.writeheader(); w.writerows(verified)
print(json.dumps({"status":"verified","pairs":len(verified),"output":str(output)}))
