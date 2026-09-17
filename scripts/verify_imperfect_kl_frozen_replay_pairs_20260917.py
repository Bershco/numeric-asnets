#!/usr/bin/env python3
"""Fail closed unless each KL-semantics pair used identical frozen inputs."""

import csv
import json
import math
import pathlib
import sys


def optimizer_steps(path: pathlib.Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("record_type") == "optimizer_step"]


def argmax_bits(step):
    return [
        row["target_pred_argmax_disagree_pre"]
        for batch in step["batches"]
        for row in batch["rows"]
    ]


root = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
verified = []

with manifest.open(newline="") as handle:
    for row in csv.DictReader(handle):
        domain, seed = row["domain"], row["seed"]
        legacy = optimizer_steps(root / f"{domain}_{seed}_frozen_legacy" / "first_update_audit.jsonl")
        deterministic = optimizer_steps(
            root / f"{domain}_{seed}_frozen_deterministic_current" / "first_update_audit.jsonl"
        )
        assert len(legacy) == len(deterministic) == 60
        expected_base = int(row["optimizer_rng_base_seed"])
        legacy_hashes = []
        deterministic_hashes = []
        for index, (left, right) in enumerate(zip(legacy, deterministic)):
            expected_step_seed = (expected_base + 1000003 * index) % 2147483647
            assert left["kl_current_forward"] == "training"
            assert right["kl_current_forward"] == "deterministic"
            assert left["fixed_step_rng_base_seed"] == right["fixed_step_rng_base_seed"] == expected_base
            assert left["fixed_step_rng_step_seed"] == right["fixed_step_rng_step_seed"] == expected_step_seed
            legacy_hashes.append(left["source_frozen_batch_sha256"])
            deterministic_hashes.append(right["source_frozen_batch_sha256"])
        assert legacy_hashes == deterministic_hashes
        # At step zero the network, data, targets and optimizer RNG are all
        # pre-treatment and therefore must match. Later gradients may diverge.
        assert math.isclose(
            legacy[0]["policy_gradient_l2"],
            deterministic[0]["policy_gradient_l2"],
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        assert argmax_bits(legacy[0]) == argmax_bits(deterministic[0])
        verified.append(
            {
                "pair_index": row["pair_index"],
                "domain": domain,
                "seed": seed,
                "steps": 60,
                "ordered_source_hashes_equal": True,
                "step0_policy_gradient_l2_equal": True,
                "step0_target_argmax_bits_equal": True,
                "status": "verified",
            }
        )

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=verified[0].keys())
    writer.writeheader()
    writer.writerows(verified)
print(json.dumps({"status": "verified", "pairs": len(verified), "output": str(output)}))
