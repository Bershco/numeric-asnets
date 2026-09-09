#!/usr/bin/env python3
"""Cache the completed Stage-1 every-five policy evaluations locally.

This is a read-only cluster extraction.  The resulting tidy CSV lets advisor
figures be rebuilt without reading 1,980 remote evaluation logs again.
"""

from __future__ import annotations

import base64
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_tracking" / "learning_curves" / "stage1_policy_curve_rows_20260909.csv"
SSH = [
    r"C:\Windows\System32\OpenSSH\ssh.exe",
    "-F",
    r"C:\Users\roeeh\.ssh\config",
    "uni-cluster",
]


def main() -> None:
    remote = r'''
import json,re
from pathlib import Path

root=Path('/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_policy_eval')
name_re=re.compile(r'^(?P<job>\d+)_Ev_(?P<domain>block_grouping|drone|fo_counters|rover|counters)_.*?_orig_(?P<vh>novh|vh)_c.*?_s(?P<seed>\d+)_SR10P_src(?P<src>\d+)_e(?P<epoch>\d+)\.txt$')
score_re=re.compile(r'\[EVAL FINAL\]\s+success=([0-9.]+)/([0-9.]+)')
rows=[]
for path in sorted(root.glob('*.txt')):
    m=name_re.match(path.name)
    if not m:
        continue
    text=path.read_text(errors='replace')
    scores=score_re.findall(text)
    if not scores:
        continue
    success,total=scores[-1]
    rows.append({
        'evaluation_job_id':m.group('job'),
        'source_training_job_id':m.group('src'),
        'domain':m.group('domain'),
        'value_head':'off' if m.group('vh')=='novh' else 'on',
        'seed':m.group('seed'),
        'epoch':int(m.group('epoch')),
        'successes':float(success),
        'total':float(total),
        'source_evaluation_log':str(path),
    })
print(json.dumps(rows,separators=(',',':')))
'''
    payload = base64.b64encode(remote.encode()).decode()
    command = f"python3 -c \"import base64;exec(base64.b64decode('{payload}'))\""
    result = subprocess.run(SSH + [command], check=True, text=True, capture_output=True)
    rows = json.loads(result.stdout)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(OUT)}))


if __name__ == "__main__":
    main()
