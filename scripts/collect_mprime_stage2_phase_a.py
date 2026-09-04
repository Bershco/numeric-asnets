#!/usr/bin/env python3
"""Consolidate completed MPrime Stage-2 validation/test checkpoint evidence."""

from __future__ import annotations

import csv
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking" / "mprime_validation_ipc_scale_v1"
MANIFESTS = {
    "validation_led": (
        TRACKING / "validation_led_stage2_policy_ready_off.csv",
        TRACKING / "validation_led_stage2_policy_ready_on.csv",
    ),
    "terminal_led": (TRACKING / "terminal_led_stage2_policy_ready.csv",),
}
ROWS_OUT = TRACKING / "validation_adequacy_phase_a_stage2_checkpoints_20260903.csv"
SEEDS_OUT = TRACKING / "validation_adequacy_phase_a_stage2_seeds_20260903.csv"
SUMMARY_OUT = TRACKING / "validation_adequacy_phase_a_stage2_summary_20260903.csv"
MISSING_OUT = TRACKING / "validation_adequacy_phase_a_stage2_missing_20260903.csv"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
CONFIG = r"C:\Users\roeeh\.ssh\config"

JOB_EPOCH_RE = re.compile(r"/(\d+)_.*_src(\d+)_e(\d{4})\.txt:")
SUCCESS_RE = re.compile(r"Inference success rate:\s*([0-9.]+)")
VAL_RE = re.compile(r"\[VAL\] Evaluator successes\s*:\s*(\d+)")
TRAINING_JOB_RE = re.compile(r"/(\d+)_.*\.txt:")
VALIDATION_RE = re.compile(r"Current network validation success rate:\s*([0-9.]+)")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def remote(command: str) -> str:
    return subprocess.check_output(
        [SSH, "-F", CONFIG, "-o", "BatchMode=yes", "uni-cluster", command],
        text=True, encoding="utf-8", errors="replace",
    )


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x-left_mean)*(y-right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x-left_mean)**2 for x in left) * sum((y-right_mean)**2 for y in right)
    )
    return numerator / denominator if denominator else float("nan")


def kendall_tau_b(left: list[float], right: list[float]) -> float:
    concordant = discordant = tied_left = tied_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = (left[i] > left[j]) - (left[i] < left[j])
            dy = (right[i] > right[j]) - (right[i] < right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_left += 1
            elif dy == 0:
                tied_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_left)
        * (concordant + discordant + tied_right)
    )
    return (concordant-discordant)/denominator if denominator else float("nan")


def main() -> None:
    evaluation_text = ""
    for date, directory in (
        ("2026-09-02", "mprime_terminal_led_stage2_policy_mprime"),
        ("2026-09-03", "mprime_terminal_led_stage2_policy_mprime"),
        ("2026-09-02", "mprime_validation_led_stage2_policy_mprime"),
    ):
        evaluation_text += remote(
            "grep -a -H -E 'Inference success rate:|\\[VAL\\] Evaluator successes' "
            f"/home/hersco/training_new_domains/{date}/{directory}/*.txt"
        )
        print(f"retrieved evaluation summaries: {date}/{directory}", flush=True)
    scores: dict[tuple[str, int], dict[str, object]] = defaultdict(dict)
    for line in evaluation_text.splitlines():
        match = JOB_EPOCH_RE.search(line)
        if not match:
            continue
        job_id, source_job, epoch_text = match.groups()
        key = (source_job, int(epoch_text))
        scores[key]["job_id"] = job_id
        scores[key]["log"] = line.split(":", 1)[0]
        success = SUCCESS_RE.search(line)
        valid = VAL_RE.search(line)
        if success:
            scores[key]["test_score"] = round(float(success.group(1)) * 20)
        if valid:
            scores[key]["val_valid"] = int(valid.group(1))

    training_text = ""
    for date, directory in (
        ("2026-09-01", "mprime_validation_led_stage2"),
        ("2026-09-01", "mprime_terminal_led_stage2"),
        ("2026-08-29", "mprime_corrected_anchor_tuning"),
    ):
        training_text += remote(
            "grep -a -H '\\[VALIDATION\\] Current network validation success rate:' "
            f"/home/hersco/training_new_domains/{date}/{directory}/*.txt"
        )
        print(f"retrieved validation summaries: {directory}", flush=True)
    validation: dict[str, list[float]] = defaultdict(list)
    training_logs: dict[str, str] = {}
    for line in training_text.splitlines():
        job = TRAINING_JOB_RE.search(line)
        value = VALIDATION_RE.search(line)
        if job and value:
            job_id = job.group(1)
            validation[job_id].append(float(value.group(1)))
            training_logs[job_id] = line.split(":", 1)[0]

    rows = []
    missing_rows = []
    for branch, paths in MANIFESTS.items():
        for path in paths:
            for row in read(path):
                source_job = row["source_training_job_id"]
                epoch = int(row["snapshot_epoch"])
                score = scores.get((source_job, epoch), {})
                if "test_score" not in score:
                    missing_rows.append({
                        "branch": branch, "value_head": row["value_head"],
                        "seed": row["seed"], "training_job_id": source_job,
                        "epoch": epoch, "checkpoint": row["source_checkpoint_ref"],
                        "training_log": row["training_log"],
                        "reason": "no completed policy score found in compact log audit",
                    })
                    continue
                values = validation.get(source_job, [])
                if epoch >= len(values):
                    raise RuntimeError(
                        f"missing validation score for {source_job}/epoch{epoch}; n={len(values)}"
                    )
                rows.append({
                    "branch": branch,
                    "value_head": row["value_head"],
                    "seed": row["seed"],
                    "training_job_id": source_job,
                    "epoch": epoch,
                    "validation_success": f"{values[epoch]:.6f}",
                    "test_success": score["test_score"],
                    "roles": row["analysis_roles"],
                    "checkpoint": row["source_checkpoint_ref"],
                    "training_log": training_logs[source_job],
                    "policy_job_id": score["job_id"],
                    "policy_log": score["log"],
                    "val_valid": score.get("val_valid", ""),
                })
    if len(rows) + len(missing_rows) != 840:
        raise RuntimeError(
            f"expected 840 checkpoint identities, got {len(rows)} scored + "
            f"{len(missing_rows)} missing"
        )

    with ROWS_OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    if missing_rows:
        with MISSING_OUT.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(missing_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(missing_rows)

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["branch"]), str(row["value_head"]), str(row["seed"]))].append(row)
    seed_rows = []
    for (branch, value_head, seed), items in sorted(groups.items()):
        items.sort(key=lambda item: int(item["epoch"]))
        val = [float(item["validation_success"]) for item in items]
        test = [float(item["test_success"]) for item in items]
        maximum = max(val)
        selected = [item for item in items if "validation_selected_policy" in str(item["roles"])]
        if len(selected) != 1:
            raise RuntimeError(f"{branch}/{value_head}/{seed}: selected={len(selected)}")
        test_best = max(items, key=lambda item: (int(item["test_success"]), -int(item["epoch"])))
        seed_rows.append({
            "branch": branch, "value_head": value_head, "seed": seed,
            "checkpoints": len(items), "validation_unique_scores": len(set(val)),
            "validation_max": f"{maximum:.6f}",
            "validation_max_fraction": f"{sum(x == maximum for x in val)/len(val):.6f}",
            "spearman_validation_test": f"{pearson(ranks(val), ranks(test)):.6f}",
            "kendall_tau_b_validation_test": f"{kendall_tau_b(val, test):.6f}",
            "selected_epoch": selected[0]["epoch"],
            "selected_test_score": selected[0]["test_success"],
            "observed_test_best_epoch": test_best["epoch"],
            "observed_test_best_score": test_best["test_success"],
            "selected_test_regret": int(test_best["test_success"])-int(selected[0]["test_success"]),
            "checkpoint_ledger": str(ROWS_OUT.relative_to(ROOT)).replace("\\", "/"),
        })
    with SEEDS_OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(seed_rows)

    summary_rows = []
    for branch in ("validation_led", "terminal_led"):
        for value_head in ("off", "on"):
            subset = [r for r in seed_rows if r["branch"] == branch and r["value_head"] == value_head]
            mean = lambda field: sum(float(r[field]) for r in subset)/len(subset)
            summary_rows.append({
                "branch": branch, "value_head": value_head,
                "lineages": len(subset), "checkpoints": sum(int(r["checkpoints"]) for r in subset),
                "mean_unique_validation_scores": f"{mean('validation_unique_scores'):.3f}",
                "mean_validation_max_fraction": f"{mean('validation_max_fraction'):.6f}",
                "positive_spearman_lineages": sum(float(r["spearman_validation_test"]) > 0 for r in subset),
                "mean_within_lineage_spearman": f"{mean('spearman_validation_test'):.6f}",
                "mean_within_lineage_kendall_tau_b": f"{mean('kendall_tau_b_validation_test'):.6f}",
                "mean_selected_test_regret": f"{mean('selected_test_regret'):.3f}",
                "seed_ledger": str(SEEDS_OUT.relative_to(ROOT)).replace("\\", "/"),
            })
    with SUMMARY_OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary_rows)
    print(
        f"wrote {len(rows)} scored checkpoints, {len(missing_rows)} missing identities, "
        f"{len(seed_rows)} lineages, {len(summary_rows)} summaries"
    )


if __name__ == "__main__":
    main()
