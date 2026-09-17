#!/usr/bin/env python3
"""Verify and freeze remote primary checkpoint hashes before submission."""
import csv, hashlib, pathlib, sys

primary, new_pairs, old_summary, out_primary, out_new = map(pathlib.Path, sys.argv[1:6])
def sha(path):
    target=path/"weights.joblib" if path.is_dir() else path
    h=hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
old={(r["domain"],r["seed"]):r["checkpoint_sha256"] for r in csv.DictReader(old_summary.open(newline=""))}
def freeze(source,output):
    rows=list(csv.DictReader(source.open(newline=""))); fields=list(rows[0])+["source_checkpoint_sha256"]
    for row in rows:
        row["source_checkpoint_sha256"]=sha(pathlib.Path(row["source_checkpoint"]))
        key=(row["domain"],row["seed"])
        if row["reuse_status"]=="reuse_verified_existing_pair":
            assert old.get(key)==row["source_checkpoint_sha256"],(key,old.get(key),row["source_checkpoint_sha256"])
    with output.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return len(rows)
assert freeze(primary,out_primary)==50
assert freeze(new_pairs,out_new)==48
print("verified and froze 50 primary checkpoint hashes; two reused Counters hashes match prior frozen inputs")
