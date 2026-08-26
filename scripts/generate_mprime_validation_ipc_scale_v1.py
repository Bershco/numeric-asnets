#!/usr/bin/env python3
"""Generate and freeze a deterministic IPC-scale MPrime validation set.

The structural ranges are declared before any checkpoint is evaluated.  This
script refuses to overwrite an existing frozen set and records both parameters
and SHA-256 checksums so selection cannot silently drift between runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problem_generator.mprime.generator import generate_multiple_problems


OUTPUT = ROOT / "problems" / "numeric" / "mprime" / "validation_ipc_scale_v1"
BASE_SEED = 20260826
TIERS = {
    "easy": dict(min_foods=8, max_foods=11, min_pleasures=2,
                 max_pleasures=4, min_pains=7, max_pains=13, max_locale=9),
    "medium": dict(min_foods=12, max_foods=16, min_pleasures=3,
                   max_pleasures=5, min_pains=14, max_pains=24, max_locale=12),
    "hard": dict(min_foods=17, max_foods=22, min_pleasures=4,
                 max_pleasures=6, min_pains=25, max_pains=46, max_locale=15),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_count(text: str, object_type: str) -> int:
    marker = f"- {object_type}"
    for line in text.splitlines():
        if marker in line:
            prefix = line.split(marker, 1)[0].replace("(:objects", "")
            return len(prefix.split())
    raise ValueError(f"Missing object declaration for {object_type}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen validation set: {OUTPUT}"
        )
    OUTPUT.mkdir(parents=True)
    rows = []
    for tier_index, (tier, parameters) in enumerate(TIERS.items()):
        tier_dir = OUTPUT / tier
        tier_seed = BASE_SEED + tier_index
        generate_multiple_problems(
            output_folder=tier_dir,
            total_num_problems=10,
            # Globally unique basenames are required by the VAL log validator.
            num_prev_instances=tier_index * 10,
            seed=tier_seed,
            **parameters,
        )
        for path in sorted(tier_dir.glob("pfile*.pddl")):
            text = path.read_text(encoding="utf-8")
            rows.append({
                "version": "mprime-validation-ipc-scale-v1",
                "tier": tier,
                "filename": path.relative_to(ROOT).as_posix(),
                "generator_seed": tier_seed,
                **parameters,
                "foods": object_count(text, "food"),
                "pleasures": object_count(text, "pleasure"),
                "pains": object_count(text, "pain"),
                "eats_edges": text.count("(eats "),
                "craves_edges": text.count("(craves "),
                "sha256": sha256(path),
            })
    manifest = OUTPUT / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "version": "mprime-validation-ipc-scale-v1",
        "base_seed": BASE_SEED,
        "instances": len(rows),
        "tier_parameters": TIERS,
        "selection_independence": (
            "Ranges fixed from IPC structural scale before checkpoint scoring; "
            "test coverage is never used to regenerate or select instances."
        ),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"generated={len(rows)} manifest={manifest}")


if __name__ == "__main__":
    main()
