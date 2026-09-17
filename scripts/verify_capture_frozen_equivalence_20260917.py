#!/usr/bin/env python3
"""Verify that captured online legacy updates equal frozen legacy replay."""
import csv, hashlib, json, math, pathlib, sys

source_root, frozen_root, manifest, output = map(pathlib.Path, sys.argv[1:5])

def steps(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("record_type") == "optimizer_step"]

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def bits(step):
    return [r["target_pred_argmax_disagree_pre"] for b in step["batches"] for r in b["rows"]]

verified=[]
for row in csv.DictReader(manifest.open(newline="")):
    name=f'{row["domain"]}_{row["seed"]}'
    left=steps(source_root/f"{name}_now_legacy"/"first_update_audit.jsonl")
    right=steps(frozen_root/f"{name}_frozen_legacy"/"first_update_audit.jsonl")
    assert len(left)==len(right)==60
    for i,(a,b) in enumerate(zip(left,right)):
        assert a["fixed_step_rng_step_seed"]==b["fixed_step_rng_step_seed"]
        capture_file=pathlib.Path(a["frozen_batch_file"])
        assert sha(capture_file)==b["source_frozen_batch_sha256"]
        assert bits(a)==bits(b)
        for key in ("policy_gradient_l2","weighted_anchor_gradient_l2","gradient_l2_unclipped","gradient_l2_applied","parameter_delta_l2"):
            assert math.isclose(float(a[key]),float(b[key]),rel_tol=1e-7,abs_tol=1e-9),(name,i,key,a[key],b[key])
    left_ep=pathlib.Path((source_root/f"{name}_now_legacy"/"endpoint_path.txt").read_text().strip())/"weights.joblib"
    right_ep=pathlib.Path((frozen_root/f"{name}_frozen_legacy"/"endpoint_path.txt").read_text().strip())/"weights.joblib"
    left_weights=sorted((source_root/f"{name}_now_legacy"/"optimizer_step_weights").glob("optimizer_step_*/weights.joblib"),key=lambda p:p.parent.name)
    right_weights=sorted((frozen_root/f"{name}_frozen_legacy"/"optimizer_step_weights").glob("optimizer_step_*/weights.joblib"),key=lambda p:p.parent.name)
    assert len(left_weights)==len(right_weights)==60,(name,len(left_weights),len(right_weights))
    assert [p.parent.name for p in left_weights]==[p.parent.name for p in right_weights]
    assert [sha(p) for p in left_weights]==[sha(p) for p in right_weights],name
    verified.append({"pair_index":row["pair_index"],"domain":row["domain"],"optimizer_schedule":row["seed"],"steps":60,"all_60_exported_weight_files_equal":True,"final_checkpoint_file_sha_equal":sha(left_ep)==sha(right_ep),"status":"verified"})
with output.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=verified[0].keys()); w.writeheader(); w.writerows(verified)
print(json.dumps({"status":"capture_equals_frozen_legacy","pairs":len(verified),"all_60_exported_weight_files_equal":True,"byte_identical_final_checkpoint_files":sum(x["final_checkpoint_file_sha_equal"] for x in verified),"output":str(output)}))
