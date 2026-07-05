#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


DOMAIN_CANDIDATES = {
    "block-grouping": [
        Path("block-grouping") / "domain.pddl",
        Path("block_grouping") / "domain.pddl",
        Path("block-grouping") / "domains" / "domain.pddl",
    ],
    "delivery": [
        Path("delivery") / "domain.pddl",
        Path("delivery") / "domains" / "domain.pddl",
    ],
    "drone": [
        Path("drone") / "domain.pddl",
        Path("drone") / "domains" / "domain.pddl",
    ],
    "fo-counters": [
        Path("fo-counters") / "domain.pddl",
        Path("fo_counters") / "domain.pddl",
        Path("fo-counters") / "domains" / "domain.pddl",
    ],
    "mprime": [
        Path("mprime") / "domain.pddl",
        Path("mprime") / "domains" / "domain.pddl",
    ],
    "rover": [
        Path("rover") / "domain.pddl",
        Path("rover") / "domains" / "domain.pddl",
    ],
    "tpp": [
        Path("tpp") / "domain.pddl",
        Path("tpp") / "domains" / "domain.pddl",
    ],
    "zenotravel": [
        Path("zenotravel") / "domain.pddl",
        Path("zenotravel") / "domains" / "domain.pddl",
    ],
}


LINE_RE = re.compile(
    r"""
    ^\[
        (?P<status>[A-Z_ ]+)
    \]\s+
    (?P<domain>\S+)\s+
    (?P<difficulty>easy|medium|hard)\s+
    (?P<problem_name>\S+\.pddl)\s+
    (?:
        cfg=(?P<cfg>\S+)\s+
        to=(?P<timeout>\d+)\s+
    )?
    (?P<planner_status>\S+)\s+
    len=(?P<plan_length>\S+)
    """,
    re.VERBOSE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_domain_file(domain_root: Path, domain: str) -> Path | None:
    for candidate in DOMAIN_CANDIDATES.get(domain, []):
        path = domain_root / candidate

        if path.exists():
            return path

    domain_dir = domain_root / domain

    if domain_dir.exists():
        matches = sorted(domain_dir.rglob("domain*.pddl"))

        if matches:
            return matches[0]

    return None


def parse_stdout_line(line: str) -> dict | None:
    match = LINE_RE.search(line.strip())

    if not match:
        return None

    status = match.group("status").strip()
    domain = match.group("domain")
    difficulty = match.group("difficulty")
    problem_name = match.group("problem_name")
    planner_status = match.group("planner_status")
    plan_length_raw = match.group("plan_length")

    plan_length = ""
    if plan_length_raw not in {"None", "null", ""}:
        plan_length = plan_length_raw

    return {
        "status": status,
        "domain": domain,
        "difficulty": difficulty,
        "problem_name": problem_name,
        "planner_status": planner_status,
        "plan_length": plan_length,
        "enhsp_config": match.group("cfg") or "",
        "enhsp_timeout": match.group("timeout") or "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse audit_generated_instances_enhsp.py stdout into audit CSV."
    )

    parser.add_argument(
        "--stdout-log",
        type=Path,
        required=True,
        help="Text file containing copied/tee'd stdout from the audit script.",
    )

    parser.add_argument(
        "--generated-root",
        type=Path,
        required=True,
        help="Root of generated_validation_instances.",
    )

    parser.add_argument(
        "--domain-root",
        type=Path,
        required=True,
        help="Root containing domain PDDL files.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="CSV to write.",
    )

    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Fail if a printed pfile cannot be found under generated-root.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stdout_log = args.stdout_log.resolve()
    generated_root = args.generated_root.resolve()
    domain_root = args.domain_root.resolve()
    output_csv = args.output_csv.resolve()

    rows = []
    skipped_lines = 0

    for line in stdout_log.read_text(errors="replace").splitlines():
        parsed = parse_stdout_line(line)

        if parsed is None:
            skipped_lines += 1
            continue

        domain = parsed["domain"]
        difficulty = parsed["difficulty"]
        problem_name = parsed["problem_name"]

        problem_path = generated_root / domain / difficulty / problem_name
        domain_path = find_domain_file(domain_root, domain)

        if problem_path.exists():
            problem_sha256 = sha256_file(problem_path)
        else:
            if args.strict_files:
                raise FileNotFoundError(f"Could not find problem file: {problem_path}")
            problem_sha256 = ""

        problem_key = f"{domain}/{difficulty}/{problem_name}"

        status = parsed["status"]
        planner_status = parsed["planner_status"]

        static_ok = status not in {"STATIC_FAIL"}
        planner_ran = status not in {"STATIC_OK", "NO_DOMAIN_FILE", "SKIPPED_SOLVED"}
        planner_success = status in {"SOLVED", "SKIPPED_SOLVED"}

        detail_parts = [
            f"parsed_from_stdout=true",
            f"enhsp_config={parsed['enhsp_config']}",
            f"enhsp_timeout={parsed['enhsp_timeout']}",
        ]

        rows.append(
            {
                "domain": domain,
                "difficulty": difficulty,
                "problem_key": problem_key,
                "problem_sha256": problem_sha256,
                "problem_path": str(problem_path),
                "domain_path": "" if domain_path is None else str(domain_path),
                "static_ok": static_ok,
                "planner_ran": planner_ran,
                "planner_success": planner_success,
                "planner_status": planner_status,
                "plan_length": parsed["plan_length"],
                "status": status,
                "detail": "; ".join(detail_parts),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "domain",
                "difficulty",
                "problem_key",
                "problem_sha256",
                "problem_path",
                "domain_path",
                "static_ok",
                "planner_ran",
                "planner_success",
                "planner_status",
                "plan_length",
                "status",
                "detail",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    counts = {}

    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    print("===== PARSED STDOUT SUMMARY =====")
    for status in sorted(counts):
        print(f"{status:15s}: {counts[status]}")

    print(f"Parsed rows   : {len(rows)}")
    print(f"Skipped lines : {skipped_lines}")
    print(f"Wrote CSV     : {output_csv}")


if __name__ == "__main__":
    main()