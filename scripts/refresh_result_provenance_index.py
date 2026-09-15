#!/usr/bin/env python3
"""Refresh only the CSV/provenance inventory, without rewriting status pages."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "result_csv_provenance_index_latest.csv"
PROVENANCE_TOKENS = ("job", "log", "path", "checkpoint", "manifest", "source", "provenance", "ledger", "evidence")
RESULT_TOKENS = ("score", "success", "coverage", "effect", "mean", "p_value", "raw_p", "holm_p", "runtime", "regret")


paths = sorted(TRACK.rglob("*.csv"))
rows: list[dict[str, object]] = []
duplicates: dict[str, list[str]] = defaultdict(list)
for path in paths:
    rel = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    duplicates[digest].append(rel)
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

# Keep the registry-path and byte-duplicate companion audits synchronized with
# the same filesystem snapshot.  The older all-purpose publication script also
# writes these files, but it can rewrite narrative status pages and is therefore
# not appropriate for a provenance-only refresh.
registry_path = TRACK / "experiment_registry.csv"
with registry_path.open(newline="", encoding="utf-8-sig") as handle:
    registry = list(csv.DictReader(handle))
reference_rows: list[dict[str, str]] = []
for row in registry:
    for field in ("results_file", "manifest_path"):
        refs = [part.strip() for part in row.get(field, "").split(";") if part.strip()]
        if not refs:
            reference_rows.append({
                "experiment_id": row["experiment_id"], "field": field,
                "reference": "", "state": "not_declared",
            })
        for ref in refs:
            if ref.startswith(("http://", "https://", "/home/")):
                state = "remote_reference"
            else:
                state = "exists_local" if (ROOT / ref).exists() else "missing_local"
            reference_rows.append({
                "experiment_id": row["experiment_id"], "field": field,
                "reference": ref, "state": state,
            })
reference_out = TRACK / "registry_reference_audit_latest.csv"
with reference_out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["experiment_id", "field", "reference", "state"])
    writer.writeheader()
    writer.writerows(reference_rows)

duplicate_rows: list[dict[str, object]] = []
for digest, rels in duplicates.items():
    if len(rels) < 2:
        continue
    registry_mirror = set(rels) == {
        "experiment_tracking/experiment_registry.csv",
        "experiment_tracking/experiments.csv",
    }
    duplicate_rows.append({
        "sha256": digest,
        "copies": len(rels),
        "paths": ";".join(rels),
        "disposition": (
            "experiments.csv is a generated compatibility mirror; edit only experiment_registry.csv"
            if registry_mirror else
            "retain if dated snapshot; canonical readers must use the *_latest or master path"
        ),
    })
duplicate_out = TRACK / "documentation_redundancy_audit_latest.csv"
with duplicate_out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=["sha256", "copies", "paths", "disposition"]
    )
    writer.writeheader()
    writer.writerows(duplicate_rows)
print(f"{reference_out}: {len(reference_rows)} references")
print(f"{duplicate_out}: {len(duplicate_rows)} duplicate groups")
