#!/usr/bin/env python3
"""Join eight realized state manifests to V1 candidates and run strict gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "asnets"))
from asnets.value_head_audit_manifest import (  # noqa: E402
    validate_label_source_rows,
    validate_state_mixture_rows,
    validate_task_manifest_rows,
)


DOMAINS = ("drone", "fo_counters", "rover", "mprime")
SEEDS = ("534933607", "923500475")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ledger: dict[tuple[str, str], dict[str, object]] = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            metas = list((args.capture_root / f"{domain}_{seed}" / "v1_states").glob("*.meta.json"))
            if len(metas) != 1:
                raise RuntimeError(f"{domain}/{seed}: expected one metadata sidecar, got {len(metas)}")
            meta = json.loads(metas[0].read_text(encoding="utf-8"))
            manifest = Path(meta["manifest_path"])
            if int(meta["states"]) != 60 or not manifest.is_file():
                raise RuntimeError(f"{domain}/{seed}: incomplete manifest")
            ledger[(domain, seed)] = meta

    base = ROOT / "experiment_tracking/value_head_quality_audit"
    candidates = read_csv(base / "v1_checkpoint_candidates.csv")
    for row in candidates:
        meta = ledger[(row["domain"], row["seed"])]
        row["state_manifest_path"] = str(meta["manifest_path"])
        row["state_manifest_sha256"] = str(meta["manifest_sha256"])
        row["status"] = "frozen_inputs_verified_resource_preflight_pending"

    errors = validate_task_manifest_rows(candidates, domains=DOMAINS, seeds=SEEDS)
    errors.extend(validate_state_mixture_rows(read_csv(base / "v1_state_mixture.csv"), domains=DOMAINS))
    errors.extend(validate_label_source_rows(read_csv(base / "v1_label_sources.csv")))
    if errors:
        raise RuntimeError("strict preflight failed:\n- " + "\n- ".join(errors))

    ready_path = args.output_dir / "v1_checkpoint_candidates.ready.csv"
    with ready_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidates[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates)
    ledger_path = args.output_dir / "v1_state_manifest_hashes.csv"
    with ledger_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("domain", "seed", "states", "manifest_path", "manifest_sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for key in sorted(ledger):
            # Metadata sidecars may add provenance fields over time.  The
            # compact hash ledger intentionally exports only its declared
            # stable schema; extra sidecar fields remain in the sidecars.
            writer.writerow({field: ledger[key][field] for field in writer.fieldnames})
    print(f"READY_INPUTS|16|8|{ready_path}|{ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
