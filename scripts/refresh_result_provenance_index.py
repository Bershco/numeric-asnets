#!/usr/bin/env python3
"""Refresh only the CSV/provenance inventory, without rewriting status pages."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "result_csv_provenance_index_latest.csv"
PROVENANCE_TOKENS = ("job", "log", "path", "checkpoint", "manifest", "source", "provenance", "ledger")
RESULT_TOKENS = ("score", "success", "coverage", "effect", "mean", "p_value", "raw_p", "holm_p", "runtime", "regret")


paths = sorted(TRACK.rglob("*.csv"))
rows: list[dict[str, object]] = []
for path in paths:
    rel = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            values = list(reader)
        parse_state = "ok"
    except Exception as exc:  # pragma: no cover - audit path
        header, values, parse_state = [], [], f"parse_error:{type(exc).__name__}"
    lower = [field.lower() for field in header]
    result_like = any(any(token in field for token in RESULT_TOKENS) for field in lower)
    provenance_fields = [field for field in header if any(token in field.lower() for token in PROVENANCE_TOKENS)]
    populated = sum(any(str(row.get(field, "")).strip() for field in provenance_fields) for row in values)
    current = "latest" in path.name or "advisor_followup_20260910" in rel
    if provenance_fields and populated == len(values):
        provenance_status = "inline_or_pointer_present"
    elif provenance_fields:
        provenance_status = "partial_or_empty_provenance"
    elif not result_like:
        provenance_status = "not_a_result_table"
    else:
        provenance_status = "legacy_gap_not_primary"
    rows.append({
        "csv_path": rel,
        "rows": len(values),
        "classification": "current_canonical" if current else "historical_or_supporting",
        "result_like": int(result_like),
        "provenance_fields": ";".join(provenance_fields),
        "populated_provenance_rows": populated,
        "provenance_status": provenance_status,
        "parse_state": parse_state,
        "sha256": digest,
    })

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{OUT}: {len(rows)} CSV files")
