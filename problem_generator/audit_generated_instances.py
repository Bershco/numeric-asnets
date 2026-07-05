#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DIFFICULTIES = ("easy", "medium", "hard")


DOMAINS = {
    "block-grouping": {
        "pddl_name": "mt-block-grouping",
        "domain_candidates": [
            Path("block-grouping") / "domain.pddl",
            Path("block-grouping") / "domains" / "domain.pddl",
        ],
    },
    "delivery": {
        "pddl_name": "delivery",
        "domain_candidates": [
            Path("delivery") / "domain.pddl",
            Path("delivery") / "domains" / "domain.pddl",
        ],
    },
    "drone": {
        "pddl_name": "drone",
        "domain_candidates": [
            Path("drone") / "domain.pddl",
            Path("drone") / "domains" / "domain.pddl",
        ],
    },
    "fo-counters": {
        "pddl_name": "fo-counters",
        "domain_candidates": [
            Path("fo-counters") / "domain.pddl",
            Path("fo-counters") / "domains" / "domain.pddl",
        ],
    },
    "mprime": {
        "pddl_name": "mystery-prime-typed",
        "domain_candidates": [
            Path("mprime") / "domain.pddl",
            Path("mprime") / "domains" / "domain.pddl",
        ],
    },
    "rover": {
        "pddl_name": "rover",
        "domain_candidates": [
            Path("rover") / "domain.pddl",
            Path("rover") / "domains" / "domain.pddl",
        ],
    },
    "tpp": {
        "pddl_name": "TPP-Metric",
        "domain_candidates": [
            Path("tpp") / "domain.pddl",
            Path("tpp") / "domains" / "domain.pddl",
        ],
    },
    "zenotravel": {
        "pddl_name": "zenotravel",
        "domain_candidates": [
            Path("zenotravel") / "domain.pddl",
            Path("zenotravel") / "domains" / "domain.pddl",
        ],
    },
}


@dataclass
class AuditResult:
    domain: str
    difficulty: str
    problem_path: Path
    domain_path: Path | None
    static_ok: bool
    runner_ok: bool | None
    status: str
    detail: str


def strip_comments(text: str) -> str:
    lines = []

    for line in text.splitlines():
        if ";" in line:
            line = line.split(";", 1)[0]
        lines.append(line)

    return "\n".join(lines)


def parentheses_balanced(text: str) -> bool:
    depth = 0

    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1

        if depth < 0:
            return False

    return depth == 0


def extract_problem_domain(text: str) -> str | None:
    match = re.search(r"\(:domain\s+([^\s()]+)\s*\)", text, flags=re.IGNORECASE)

    if not match:
        return None

    return match.group(1)


def find_domain_file(problem_generator_root: Path, domain_name: str) -> Path | None:
    for candidate in DOMAINS[domain_name]["domain_candidates"]:
        path = problem_generator_root / candidate

        if path.exists():
            return path

    # Last-resort fallback: search within that domain folder.
    domain_dir = problem_generator_root / domain_name

    if domain_dir.exists():
        matches = sorted(domain_dir.rglob("domain*.pddl"))

        if matches:
            return matches[0]

    return None


def static_check_problem(
    *,
    problem_path: Path,
    expected_pddl_domain: str,
) -> tuple[bool, str]:
    try:
        raw_text = problem_path.read_text(errors="replace")
    except Exception as exc:
        return False, f"could not read file: {exc}"

    text = strip_comments(raw_text)

    if "$" in text:
        return False, "file contains '$', likely unreplaced template variable"

    if not parentheses_balanced(text):
        return False, "unbalanced parentheses"

    if not re.search(r"\(define\s+\(problem\s+", text, flags=re.IGNORECASE):
        return False, "missing '(define (problem ...)'"

    actual_domain = extract_problem_domain(text)

    if actual_domain is None:
        return False, "missing (:domain ...)"

    if actual_domain != expected_pddl_domain:
        return (
            False,
            f"wrong domain: expected {expected_pddl_domain!r}, got {actual_domain!r}",
        )

    for section in (":objects", ":init", ":goal"):
        if section not in text:
            return False, f"missing {section} section"

    return True, "static ok"


def run_external_check(
    *,
    cmd_template: str,
    domain_path: Path,
    problem_path: Path,
    domain: str,
    difficulty: str,
    timeout_seconds: int,
    success_regex: str | None,
) -> tuple[bool, str]:
    command = cmd_template.format(
        domain_pddl=shlex.quote(str(domain_path)),
        problem_pddl=shlex.quote(str(problem_path)),
        domain=shlex.quote(domain),
        difficulty=shlex.quote(difficulty),
        problem_name=shlex.quote(problem_path.name),
    )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds}s"

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")

    if success_regex is not None:
        if re.search(success_regex, output, flags=re.IGNORECASE | re.MULTILINE):
            return True, "runner success regex matched"

        tail = "\n".join(output.splitlines()[-20:])
        return False, f"success regex did not match; exit={completed.returncode}; tail:\n{tail}"

    if completed.returncode == 0:
        return True, "runner exit code 0"

    tail = "\n".join(output.splitlines()[-20:])
    return False, f"runner exit={completed.returncode}; tail:\n{tail}"


def audit_instance(
    *,
    problem_generator_root: Path,
    generated_root: Path,
    domain: str,
    difficulty: str,
    problem_path: Path,
    cmd_template: str | None,
    timeout_seconds: int,
    success_regex: str | None,
) -> AuditResult:
    expected_pddl_domain = DOMAINS[domain]["pddl_name"]
    domain_path = find_domain_file(problem_generator_root, domain)

    static_ok, static_detail = static_check_problem(
        problem_path=problem_path,
        expected_pddl_domain=expected_pddl_domain,
    )

    if not static_ok:
        return AuditResult(
            domain=domain,
            difficulty=difficulty,
            problem_path=problem_path,
            domain_path=domain_path,
            static_ok=False,
            runner_ok=None,
            status="STATIC_FAIL",
            detail=static_detail,
        )

    if cmd_template is None:
        return AuditResult(
            domain=domain,
            difficulty=difficulty,
            problem_path=problem_path,
            domain_path=domain_path,
            static_ok=True,
            runner_ok=None,
            status="STATIC_OK",
            detail=static_detail,
        )

    if domain_path is None:
        return AuditResult(
            domain=domain,
            difficulty=difficulty,
            problem_path=problem_path,
            domain_path=None,
            static_ok=True,
            runner_ok=False,
            status="NO_DOMAIN_FILE",
            detail="could not locate domain.pddl",
        )

    runner_ok, runner_detail = run_external_check(
        cmd_template=cmd_template,
        domain_path=domain_path,
        problem_path=problem_path,
        domain=domain,
        difficulty=difficulty,
        timeout_seconds=timeout_seconds,
        success_regex=success_regex,
    )

    return AuditResult(
        domain=domain,
        difficulty=difficulty,
        problem_path=problem_path,
        domain_path=domain_path,
        static_ok=True,
        runner_ok=runner_ok,
        status="RUNNER_OK" if runner_ok else "RUNNER_FAIL",
        detail=runner_detail,
    )


def write_csv(results: list[AuditResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "domain",
                "difficulty",
                "problem_path",
                "domain_path",
                "static_ok",
                "runner_ok",
                "status",
                "detail",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "domain": result.domain,
                    "difficulty": result.difficulty,
                    "problem_path": str(result.problem_path),
                    "domain_path": "" if result.domain_path is None else str(result.domain_path),
                    "static_ok": result.static_ok,
                    "runner_ok": "" if result.runner_ok is None else result.runner_ok,
                    "status": result.status,
                    "detail": result.detail,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit generated validation PDDL instances."
    )

    parser.add_argument(
        "--problem-generator-root",
        type=Path,
        required=True,
        help="Path to problem_generator.",
    )

    parser.add_argument(
        "--generated-root",
        type=Path,
        required=True,
        help="Path to generated_validation_instances.",
    )

    parser.add_argument(
        "--expected-per-difficulty",
        type=int,
        default=10,
        help="Expected number of .pddl files per domain/difficulty. Default: 10.",
    )

    parser.add_argument(
        "--cmd-template",
        default=None,
        help=(
            "Optional external command to run for each instance. "
            "Available placeholders: {domain_pddl}, {problem_pddl}, "
            "{domain}, {difficulty}, {problem_name}. "
            "Paths are shell-quoted automatically."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="Timeout per external command. Default: 20.",
    )

    parser.add_argument(
        "--success-regex",
        default=None,
        help=(
            "Optional regex that must match stdout/stderr for runner success. "
            "If omitted, exit code 0 is considered success."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("generated_instances_audit.csv"),
        help="CSV output path. Default: generated_instances_audit.csv.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    problem_generator_root = args.problem_generator_root.resolve()
    generated_root = args.generated_root.resolve()

    results = []
    structural_errors = []

    for domain in DOMAINS:
        for difficulty in DIFFICULTIES:
            folder = generated_root / domain / difficulty

            if not folder.exists():
                structural_errors.append(f"missing folder: {folder}")
                continue

            problems = sorted(folder.glob("*.pddl"))

            if len(problems) != args.expected_per_difficulty:
                structural_errors.append(
                    f"{domain}/{difficulty}: expected "
                    f"{args.expected_per_difficulty} files, found {len(problems)}"
                )

            for problem_path in problems:
                result = audit_instance(
                    problem_generator_root=problem_generator_root,
                    generated_root=generated_root,
                    domain=domain,
                    difficulty=difficulty,
                    problem_path=problem_path,
                    cmd_template=args.cmd_template,
                    timeout_seconds=args.timeout_seconds,
                    success_regex=args.success_regex,
                )

                results.append(result)

    write_csv(results, args.output_csv)

    counts = {}

    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print("===== AUDIT SUMMARY =====")

    for key in sorted(counts):
        print(f"{key:15s}: {counts[key]}")

    if structural_errors:
        print()
        print("===== STRUCTURAL ERRORS =====")

        for error in structural_errors:
            print(error)

    bad_results = [
        result
        for result in results
        if result.status not in {"STATIC_OK", "RUNNER_OK"}
    ]

    if bad_results:
        print()
        print("===== BAD INSTANCES =====")

        for result in bad_results[:50]:
            print(
                f"{result.status:15s} {result.domain}/{result.difficulty} "
                f"{result.problem_path}: {result.detail}"
            )

        if len(bad_results) > 50:
            print(f"... and {len(bad_results) - 50} more")

    print()
    print(f"Wrote CSV: {args.output_csv}")

    if structural_errors or bad_results:
        sys.exit(1)


if __name__ == "__main__":
    main()