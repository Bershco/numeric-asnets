#!/usr/bin/env python3
"""Select one exact captured state per frozen V1 instance/source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "asnets"))
from asnets.value_head_audit import validate_canonical_state_record  # noqa: E402


def expand(spec: str) -> list[str]:
    result: list[str] = []
    for part in spec.split(";"):
        tier, expr = part.split(":", 1)
        match = re.fullmatch(r"pfile(\d+)-(\d+)", expr)
        if not match:
            raise ValueError(f"invalid frozen instance range: {part}")
        start, stop = map(int, match.groups())
        result.extend(f"{tier}/pfile{i}.pddl" for i in range(start, stop + 1))
    return result


def instance_identity(path: str) -> str:
    normalized = path.replace("\\", "/")
    match = re.search(
        r"(?:validation_ipc_scale_v1/)?(?:valid_)?(easy|medium|hard)/(pfile\d+\.pddl)$",
        normalized,
    )
    if not match:
        raise ValueError(f"cannot normalize captured instance path: {path}")
    return f"{match.group(1)}/{match.group(2)}"


def read_captures(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                validate_canonical_state_record(record)
                records.append(record)
    return records


def select_balanced_unique(
    *,
    by_instance: dict[str, list[dict[str, object]]],
    expected: list[str],
    quota: int,
    rank,
    seen_hashes: set[str],
) -> list[tuple[str, dict[str, object], str]]:
    """Select deterministically without requiring every actor to reach every instance."""
    ranked = {
        ident: sorted(by_instance.get(ident, []), key=lambda row: rank(ident, row))
        for ident in expected
    }
    selected: list[tuple[str, dict[str, object], str]] = []
    depth = 0
    while len(selected) < quota:
        added = False
        for ident in expected:
            candidates = ranked[ident]
            while depth < len(candidates):
                chosen = candidates[depth]
                state_hash = str(chosen["state_sha256"])
                if state_hash not in seen_hashes:
                    seen_hashes.add(state_hash)
                    selected.append((ident, chosen, rank(ident, chosen)))
                    added = True
                    break
                candidates.pop(depth)
            if len(selected) == quota:
                break
        if not added:
            break
        depth += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    mixture_path = ROOT / "experiment_tracking/value_head_quality_audit/v1_state_mixture.csv"
    with mixture_path.open(newline="", encoding="utf-8") as stream:
        mixture = [
            row for row in csv.DictReader(stream) if row["domain"] == args.domain
        ]
    if len(mixture) != 4:
        raise RuntimeError(f"expected four source rows for {args.domain}")

    selected: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    for source_row in mixture:
        source = source_row["state_source"]
        expected = expand(source_row["validation_instances"])
        quota = int(source_row["quota_per_lineage_manifest"])
        if len(expected) != quota:
            raise RuntimeError(f"{source}: expected-instance count != quota")
        by_instance: dict[str, list[dict[str, object]]] = {}
        for record in read_captures(args.capture_root / source):
            if record.get("state_source") != source:
                raise RuntimeError(f"{source}: capture source mismatch")
            if int(record.get("step", 0)) <= 0 or record.get("is_terminal"):
                continue
            ident = instance_identity(str(record["instance_path"]))
            by_instance.setdefault(ident, []).append(record)
        def rank(ident: str, record: dict[str, object]) -> str:
            payload = (
                f"{args.domain}|{args.seed}|{source}|{ident}|"
                f"{record['state_sha256']}"
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # Actor feasibility is independent of learned-value outcomes.  Some
        # frozen instances yield no post-initial state (for example, ENHSP can
        # time out before producing a plan).  Fill the frozen source quota in
        # balanced rounds over the predeclared instance pool, rather than
        # selecting replacement instances after observing network values.
        chosen_rows = select_balanced_unique(
            by_instance=by_instance,
            expected=expected,
            quota=quota,
            rank=rank,
            seen_hashes=seen_hashes,
        )
        if len(chosen_rows) != quota:
            represented = sum(bool(by_instance.get(ident)) for ident in expected)
            raise RuntimeError(
                f"{source}: selected {len(chosen_rows)}/{quota} unique "
                f"post-initial states from {represented}/{len(expected)} frozen instances"
            )
        selected.extend({
            **chosen,
            "audit_domain": args.domain,
            "audit_seed": int(args.seed),
            "instance_identity": ident,
            "selection_rank_sha256": selection_rank,
        } for ident, chosen, selection_rank in chosen_rows)

    if len(selected) != 60:
        raise RuntimeError(f"expected 60 selected states, got {len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in selected:
            stream.write(json.dumps(
                record, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ) + "\n")
    temporary.replace(args.output)
    manifest_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    sidecar = args.output.with_suffix(args.output.suffix + ".meta.json")
    sidecar.write_text(json.dumps({
        "schema": "value-head-audit-state-manifest-v1",
        "domain": args.domain,
        "seed": int(args.seed),
        "states": len(selected),
        "manifest_path": str(args.output),
        "manifest_sha256": manifest_hash,
        "represented_instances_by_source": {
            source: len({
                str(row["instance_identity"])
                for row in selected if row["state_source"] == source
            })
            for source in sorted({str(row["state_source"]) for row in selected})
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MATERIALIZED|{args.domain}|{args.seed}|60|{manifest_hash}|{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
