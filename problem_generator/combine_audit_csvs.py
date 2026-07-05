#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PFILE_RE = re.compile(r"^pfile(\d+)\.pddl$")
TIMEOUT_RE = re.compile(
    r"(?:enhsp_timeout|timeout|timeout_seconds)\s*=\s*(\d+)",
    flags=re.IGNORECASE,
)
CONFIG_RE = re.compile(
    r"(?:enhsp_config|config|cfg)\s*=\s*([^;,\s]+)",
    flags=re.IGNORECASE,
)


# These are based on the audit runs you described.
# If you know a filename used a different timeout, edit only this dict.
SOURCE_TIMEOUT_HINTS: dict[str, dict[str, int] | int] = {
    # Parsed stdout CSV usually already has enhsp_timeout in detail.
    # This fallback is only used if that detail is missing.
    "audit_from_stdout.csv": {
        "block-grouping": 30,
        "delivery": 30,
        "drone": 30,
        "fo-counters": 30,
        "mprime": 30,
        "rover": 60,
        "tpp": 60,
        "zenotravel": 60,
    },

    # First retry with domain configs.
    "audit_retry.csv": {
        "block-grouping": 30,
        "delivery": 30,
        "drone": 30,
        "fo-counters": 30,
        "mprime": 30,
        "rover": 60,
        "tpp": 60,
        "zenotravel": 60,
    },

    # Longer-timeout pass.
    "audit_retry_longer.csv": {
        "block-grouping": 120,
        "delivery": 180,
        "drone": 120,
        "fo-counters": 120,
        "mprime": 30,
        "rover": 180,
        "tpp": 180,
        "zenotravel": 120,
    },

    # After failed-instance regeneration.
    "audit_after_regen.csv": {
        "block-grouping": 120,
        "delivery": 180,
        "drone": 120,
        "fo-counters": 120,
        "mprime": 30,
        "rover": 180,
        "tpp": 180,
        "zenotravel": 120,
    },

    # Later config/timeout pass.
    "audit_after_zenotravel_harder.csv": {
        "block-grouping": 120,
        "delivery": 300,
        "drone": 120,
        "fo-counters": 180,
        "mprime": 30,
        "rover": 180,
        "tpp": 300,
        "zenotravel": 120,
    },

    # Targeted reduced TPP hard pass.
    "audit_tpp_hard_reduced.csv": {
        "tpp": 300,
    },

    # Targeted middle-ground Zenotravel hard pass.
    "audit_zenotravel_hard_mid.csv": {
        "zenotravel": 120,
    },
}


# These are not useful as final solve/fail evidence.
DEFAULT_IGNORE_SOURCE_NAMES = {
    "audit_static.csv",
    "audit_enhsp_15s.csv",
    "all_combined.csv",
}


SOLVED_STATUSES = {"SOLVED"}
SKIPPED_SOLVED_STATUSES = {"SKIPPED_SOLVED"}
FAIL_STATUSES = {"PLANNER_FAIL", "STATIC_FAIL", "NO_DOMAIN_FILE"}


@dataclass(frozen=True)
class CurrentProblem:
    domain: str
    difficulty: str
    problem_name: str
    problem_key: str
    problem_sha256: str
    problem_path: Path


@dataclass
class EvidenceRow:
    source_csv: str
    domain: str
    difficulty: str
    problem_name: str
    problem_key: str
    problem_sha256: str
    problem_path: str
    domain_path: str
    status: str
    planner_status: str
    planner_success: bool
    planner_ran: bool
    plan_length: int | None
    timeout_seconds: int | None
    enhsp_config: str
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def bool_from_csv(value: Any) -> bool:
    raw = str(value).strip().lower()

    return raw in {"true", "1", "yes"}


def none_if_nan(value: Any) -> str:
    if value is None:
        return ""

    raw = str(value).strip()

    if raw.lower() in {"", "nan", "none", "null"}:
        return ""

    return raw


def int_or_none(value: Any) -> int | None:
    raw = none_if_nan(value)

    if not raw:
        return None

    try:
        return int(float(raw))
    except ValueError:
        return None


def parse_problem_name(problem_path_or_key: str) -> str:
    return Path(problem_path_or_key).name


def parse_problem_key(row: dict[str, str]) -> tuple[str, str, str]:
    domain = row.get("domain", "").strip()
    difficulty = row.get("difficulty", "").strip()

    if row.get("problem_key"):
        problem_name = Path(row["problem_key"]).name
    else:
        problem_name = parse_problem_name(row.get("problem_path", ""))

    if not domain or not difficulty or not problem_name:
        raise ValueError(f"Cannot parse row identity: {row}")

    return domain, difficulty, f"{domain}/{difficulty}/{problem_name}"


def parse_timeout_from_detail(detail: str) -> int | None:
    match = TIMEOUT_RE.search(detail or "")

    if match is None:
        return None

    return int(match.group(1))


def parse_config_from_detail(detail: str) -> str:
    match = CONFIG_RE.search(detail or "")

    if match is None:
        return ""

    return match.group(1)


def timeout_hint_for_source(
    *,
    source_csv: str,
    domain: str,
) -> int | None:
    hint = SOURCE_TIMEOUT_HINTS.get(source_csv)

    if hint is None:
        return None

    if isinstance(hint, int):
        return hint

    return hint.get(domain)


def extract_timeout_seconds(
    *,
    row: dict[str, str],
    source_csv: str,
    domain: str,
) -> int | None:
    for column in (
        "enhsp_timeout",
        "timeout_seconds",
        "timeout",
        "enhsp_timeout_seconds",
    ):
        if column in row:
            value = int_or_none(row.get(column))

            if value is not None:
                return value

    detail_timeout = parse_timeout_from_detail(row.get("detail", ""))

    if detail_timeout is not None:
        return detail_timeout

    return timeout_hint_for_source(source_csv=source_csv, domain=domain)


def extract_enhsp_config(row: dict[str, str]) -> str:
    for column in ("enhsp_config", "config", "cfg"):
        if row.get(column):
            return row[column].strip()

    return parse_config_from_detail(row.get("detail", ""))


def scan_current_problems(generated_root: Path) -> dict[str, CurrentProblem]:
    current: dict[str, CurrentProblem] = {}

    for problem_path in sorted(generated_root.glob("*/*/pfile*.pddl")):
        rel = problem_path.relative_to(generated_root)

        if len(rel.parts) != 3:
            continue

        domain, difficulty, problem_name = rel.parts

        if PFILE_RE.match(problem_name) is None:
            continue

        problem_key = f"{domain}/{difficulty}/{problem_name}"

        current[problem_key] = CurrentProblem(
            domain=domain,
            difficulty=difficulty,
            problem_name=problem_name,
            problem_key=problem_key,
            problem_sha256=sha256_file(problem_path),
            problem_path=problem_path,
        )

    return current


def read_evidence_rows(
    *,
    csv_dir: Path,
    current_problems: dict[str, CurrentProblem],
    ignore_source_names: set[str],
    include_hashless_rows: bool,
) -> list[EvidenceRow]:
    evidence: list[EvidenceRow] = []

    for csv_path in sorted(csv_dir.glob("*.csv")):
        source_csv = csv_path.name

        if source_csv in ignore_source_names:
            continue

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                continue

            for row in reader:
                try:
                    domain, difficulty, problem_key = parse_problem_key(row)
                except ValueError:
                    continue

                current = current_problems.get(problem_key)

                if current is None:
                    continue

                row_hash = none_if_nan(row.get("problem_sha256"))

                if row_hash:
                    if row_hash != current.problem_sha256:
                        # Old evidence for an overwritten pfile. Ignore.
                        continue
                elif not include_hashless_rows:
                    # Hashless old rows are unsafe after regeneration.
                    continue

                status = none_if_nan(row.get("status"))
                planner_status = none_if_nan(row.get("planner_status"))

                if status not in SOLVED_STATUSES | SKIPPED_SOLVED_STATUSES | FAIL_STATUSES:
                    continue

                problem_name = current.problem_name
                detail = none_if_nan(row.get("detail"))

                evidence.append(
                    EvidenceRow(
                        source_csv=source_csv,
                        domain=domain,
                        difficulty=difficulty,
                        problem_name=problem_name,
                        problem_key=problem_key,
                        problem_sha256=current.problem_sha256,
                        problem_path=str(current.problem_path),
                        domain_path=none_if_nan(row.get("domain_path")),
                        status=status,
                        planner_status=planner_status,
                        planner_success=bool_from_csv(row.get("planner_success")),
                        planner_ran=bool_from_csv(row.get("planner_ran")),
                        plan_length=int_or_none(row.get("plan_length")),
                        timeout_seconds=extract_timeout_seconds(
                            row=row,
                            source_csv=source_csv,
                            domain=domain,
                        ),
                        enhsp_config=extract_enhsp_config(row),
                        detail=detail,
                    )
                )

    return evidence


def sort_solved_evidence(row: EvidenceRow) -> tuple[float, int, str]:
    timeout = row.timeout_seconds

    timeout_key = math.inf if timeout is None else timeout

    # If same timeout, prefer a row that actually has a length.
    has_no_length = 1 if row.plan_length is None else 0

    return timeout_key, has_no_length, row.source_csv


def sort_failed_evidence(row: EvidenceRow) -> tuple[float, str]:
    timeout = row.timeout_seconds

    timeout_key = -1 if timeout is None else timeout

    return timeout_key, row.source_csv


def combine_for_problem(
    *,
    current: CurrentProblem,
    rows: list[EvidenceRow],
) -> dict[str, Any]:
    solved_rows = [
        row
        for row in rows
        if row.status == "SOLVED"
        and (
            row.planner_success
            or row.planner_status == "SUCCESS"
        )
    ]

    fail_rows = [
        row
        for row in rows
        if row.status in FAIL_STATUSES
    ]

    skipped_rows = [
        row
        for row in rows
        if row.status in SKIPPED_SOLVED_STATUSES
    ]

    if solved_rows:
        best = sorted(solved_rows, key=sort_solved_evidence)[0]

        return {
            "domain": current.domain,
            "difficulty": current.difficulty,
            "problem_name": current.problem_name,
            "problem_key": current.problem_key,
            "problem_sha256": current.problem_sha256,
            "problem_path": str(current.problem_path),
            "final_status": "SOLVED",
            "planner_status": best.planner_status,
            "plan_length": "" if best.plan_length is None else best.plan_length,
            "selected_timeout_seconds": "" if best.timeout_seconds is None else best.timeout_seconds,
            "timeout_interpretation": "lowest_success_timeout",
            "enhsp_config": best.enhsp_config,
            "selected_source_csv": best.source_csv,
            "solved_evidence_count": len(solved_rows),
            "fail_evidence_count": len(fail_rows),
            "skipped_solved_count": len(skipped_rows),
            "all_statuses_seen": "|".join(sorted({row.status for row in rows})),
            "all_planner_statuses_seen": "|".join(sorted({row.planner_status for row in rows if row.planner_status})),
            "all_source_csvs_seen": "|".join(sorted({row.source_csv for row in rows})),
            "notes": "",
        }

    if fail_rows:
        best = sorted(fail_rows, key=sort_failed_evidence, reverse=True)[0]

        return {
            "domain": current.domain,
            "difficulty": current.difficulty,
            "problem_name": current.problem_name,
            "problem_key": current.problem_key,
            "problem_sha256": current.problem_sha256,
            "problem_path": str(current.problem_path),
            "final_status": "UNSOLVED_BY_AUDITS",
            "planner_status": best.planner_status,
            "plan_length": "",
            "selected_timeout_seconds": "" if best.timeout_seconds is None else best.timeout_seconds,
            "timeout_interpretation": "highest_failure_timeout",
            "enhsp_config": best.enhsp_config,
            "selected_source_csv": best.source_csv,
            "solved_evidence_count": len(solved_rows),
            "fail_evidence_count": len(fail_rows),
            "skipped_solved_count": len(skipped_rows),
            "all_statuses_seen": "|".join(sorted({row.status for row in rows})),
            "all_planner_statuses_seen": "|".join(sorted({row.planner_status for row in rows if row.planner_status})),
            "all_source_csvs_seen": "|".join(sorted({row.source_csv for row in rows})),
            "notes": "",
        }

    if skipped_rows:
        return {
            "domain": current.domain,
            "difficulty": current.difficulty,
            "problem_name": current.problem_name,
            "problem_key": current.problem_key,
            "problem_sha256": current.problem_sha256,
            "problem_path": str(current.problem_path),
            "final_status": "SKIPPED_SOLVED_ONLY",
            "planner_status": "SKIPPED",
            "plan_length": "",
            "selected_timeout_seconds": "",
            "timeout_interpretation": "missing_original_solved_row",
            "enhsp_config": "",
            "selected_source_csv": "|".join(sorted({row.source_csv for row in skipped_rows})),
            "solved_evidence_count": 0,
            "fail_evidence_count": 0,
            "skipped_solved_count": len(skipped_rows),
            "all_statuses_seen": "|".join(sorted({row.status for row in rows})),
            "all_planner_statuses_seen": "|".join(sorted({row.planner_status for row in rows if row.planner_status})),
            "all_source_csvs_seen": "|".join(sorted({row.source_csv for row in rows})),
            "notes": "This hash was skipped as solved, but the original SOLVED row was not found among included CSVs.",
        }

    return {
        "domain": current.domain,
        "difficulty": current.difficulty,
        "problem_name": current.problem_name,
        "problem_key": current.problem_key,
        "problem_sha256": current.problem_sha256,
        "problem_path": str(current.problem_path),
        "final_status": "NO_AUDIT_EVIDENCE_FOR_CURRENT_HASH",
        "planner_status": "",
        "plan_length": "",
        "selected_timeout_seconds": "",
        "timeout_interpretation": "",
        "enhsp_config": "",
        "selected_source_csv": "",
        "solved_evidence_count": 0,
        "fail_evidence_count": 0,
        "skipped_solved_count": 0,
        "all_statuses_seen": "",
        "all_planner_statuses_seen": "",
        "all_source_csvs_seen": "",
        "notes": "No included CSV row matched this current file hash.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "domain",
        "difficulty",
        "problem_name",
        "problem_key",
        "problem_sha256",
        "problem_path",
        "final_status",
        "planner_status",
        "plan_length",
        "selected_timeout_seconds",
        "timeout_interpretation",
        "enhsp_config",
        "selected_source_csv",
        "solved_evidence_count",
        "fail_evidence_count",
        "skipped_solved_count",
        "all_statuses_seen",
        "all_planner_statuses_seen",
        "all_source_csvs_seen",
        "notes",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple generated-instance audit CSVs into one final hash-safe manifest."
    )

    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing audit CSVs.",
    )

    parser.add_argument(
        "--generated-root",
        type=Path,
        required=True,
        help="Current generated_validation_instances directory.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output final manifest CSV. "
            "Default: <csv-dir>/final_validation_manifest.csv"
        ),
    )

    parser.add_argument(
        "--include-hashless-rows",
        action="store_true",
        help=(
            "Use rows with no problem_sha256. Unsafe after regeneration. "
            "Default is to ignore hashless rows."
        ),
    )

    parser.add_argument(
        "--include-static",
        action="store_true",
        help="Do not ignore audit_static.csv.",
    )

    parser.add_argument(
        "--include-all-combined",
        action="store_true",
        help=(
            "Do not ignore all_combined.csv. Usually not needed because it was "
            "a timeout=1 overview, not a real evidence source."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_dir = args.csv_dir.resolve()
    generated_root = args.generated_root.resolve()
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else csv_dir / "final_validation_manifest.csv"
    )

    ignore_source_names = set(DEFAULT_IGNORE_SOURCE_NAMES)

    if args.include_static:
        ignore_source_names.discard("audit_static.csv")

    if args.include_all_combined:
        ignore_source_names.discard("all_combined.csv")

    current_problems = scan_current_problems(generated_root)

    evidence_rows = read_evidence_rows(
        csv_dir=csv_dir,
        current_problems=current_problems,
        ignore_source_names=ignore_source_names,
        include_hashless_rows=args.include_hashless_rows,
    )

    evidence_by_key: dict[str, list[EvidenceRow]] = {}

    for row in evidence_rows:
        evidence_by_key.setdefault(row.problem_key, []).append(row)

    final_rows = [
        combine_for_problem(
            current=current,
            rows=evidence_by_key.get(problem_key, []),
        )
        for problem_key, current in sorted(current_problems.items())
    ]

    write_csv(output_csv, final_rows)

    counts: dict[str, int] = {}

    for row in final_rows:
        counts[row["final_status"]] = counts.get(row["final_status"], 0) + 1

    print("===== FINAL MANIFEST SUMMARY =====")

    for status in sorted(counts):
        print(f"{status:35s}: {counts[status]}")

    print()
    print(f"Current PDDL files : {len(current_problems)}")
    print(f"Evidence rows used : {len(evidence_rows)}")
    print(f"Ignored sources    : {sorted(ignore_source_names)}")
    print(f"Wrote              : {output_csv}")


if __name__ == "__main__":
    main()