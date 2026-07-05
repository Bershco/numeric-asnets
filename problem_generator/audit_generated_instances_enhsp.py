#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib


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

DOMAIN_ENHSP_CONFIGS = {
    "block-grouping": "hadd-gbfs",
    "delivery": "hadd-gbfs",
    "drone": "hmrp-ha-ht-gbfs",
    "fo-counters": "hadd-gbfs",
    "mprime": "hmrp-ha-ht-gbfs",
    "rover": "hadd-gbfs",
    "tpp": "hadd-gbfs",
    "zenotravel": "hadd-gbfs",
}

DOMAIN_ENHSP_TIMEOUTS = {
    "block-grouping": 1,
    "delivery": 1,
    "drone": 1,
    "fo-counters": 1,
    "mprime": 1,
    "rover": 1,
    "tpp": 1,
    "zenotravel": 1,
}

@dataclass
class AuditResult:
    domain: str
    difficulty: str
    problem_key: str
    problem_sha256: str
    problem_path: Path
    domain_path: Path | None
    static_ok: bool
    planner_ran: bool
    planner_success: bool
    planner_status: str
    plan_length: int | None
    status: str
    detail: str


def configure_import_paths(project_root: Path) -> None:
    """Add likely ASNets source folders to sys.path.

    Your spawn_train_worker.py lives under:

        numeric-asnets/asnets/asnets/spawn_train_worker.py

    and imports:

        from enhsp_wrapper.enhsp import ...
        from .interfaces.enhsp_interface import ...

    For this standalone script, we support importing either:
        enhsp_wrapper.enhsp
        interfaces.enhsp_interface
    """

    candidates = [
        project_root,
        project_root / "asnets",
        project_root / "asnets" / "asnets",
    ]

    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


def load_enhsp(project_root: Path) -> tuple[Any, Any, dict[str, str]]:
    configure_import_paths(project_root)

    try:
        from enhsp_wrapper.enhsp import ENHSP, PlanningStatus
    except Exception as exc:
        raise ImportError(
            "Failed to import ENHSP / PlanningStatus from enhsp_wrapper.enhsp. "
            "Check --project-root and sys.path assumptions."
        ) from exc

    try:
        from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS
    except Exception as exc:
        raise ImportError(
            "Failed to import ENHSP_CONFIGS. Tried both:\n"
            "  from interfaces.enhsp_interface import ENHSP_CONFIGS\n"
            "  from asnets.interfaces.enhsp_interface import ENHSP_CONFIGS"
        ) from exc

    return ENHSP, PlanningStatus, ENHSP_CONFIGS


def strip_comments(text: str) -> str:
    lines = []

    for line in text.splitlines():
        if ";" in line:
            line = line.split(";", 1)[0]
        lines.append(line)

    return "\n".join(lines)

def sha256_file(path: Path) -> str:
    """Return SHA256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def make_problem_key(
    *,
    domain: str,
    difficulty: str,
    problem_path: Path,
) -> str:
    """Stable key independent of absolute machine path."""

    return f"{domain}/{difficulty}/{problem_path.name}"


def load_solved_instance_index(
    *,
    csv_paths: list[Path],
    solved_statuses: set[str],
) -> dict[str, set[str | None]]:
    """Load previously solved instances.

    Returns:
        problem_key -> set of known SHA256 hashes.

    If an older CSV has no problem_sha256 column, None is stored as a
    legacy hash marker.
    """

    solved: dict[str, set[str | None]] = {}

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"skip-solved CSV does not exist: {csv_path}")

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row.get("status") not in solved_statuses:
                    continue

                domain = row.get("domain")
                difficulty = row.get("difficulty")
                problem_path_raw = row.get("problem_path")

                if not domain or not difficulty or not problem_path_raw:
                    continue

                problem_key = f"{domain}/{difficulty}/{Path(problem_path_raw).name}"

                problem_hash = row.get("problem_sha256") or None

                solved.setdefault(problem_key, set()).add(problem_hash)

    return solved


def should_skip_solved_instance(
    *,
    problem_key: str,
    problem_sha256: str,
    solved_index: dict[str, set[str | None]],
    allow_legacy_skip_without_hash: bool,
) -> bool:
    """Return True if this exact instance was already solved."""

    known_hashes = solved_index.get(problem_key)

    if not known_hashes:
        return False

    if problem_sha256 in known_hashes:
        return True

    if allow_legacy_skip_without_hash and None in known_hashes:
        return True

    return False


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

    if match is None:
        return None

    return match.group(1)


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


def find_domain_file(
    *,
    problem_generator_root: Path,
    domain_root: Path,
    domain_name: str,
) -> Path | None:
    spec = DOMAINS[domain_name]

    search_roots = [
        domain_root,
        problem_generator_root,
    ]

    for root in search_roots:
        for candidate in spec["domain_candidates"]:
            path = root / candidate

            if path.exists():
                return path

        domain_dir = root / domain_name

        if domain_dir.exists():
            matches = sorted(domain_dir.rglob("domain*.pddl"))

            if matches:
                return matches[0]

    return None


def run_enhsp_planner(
    *,
    ENHSP: Any,
    PlanningStatus: Any,
    ENHSP_CONFIGS: dict[str, str],
    enhsp_config: str,
    enhsp_timeout: int,
    domain_path: Path,
    problem_path: Path,
) -> tuple[bool, str, int | None, str]:
    if enhsp_config not in ENHSP_CONFIGS:
        known = ", ".join(sorted(ENHSP_CONFIGS.keys()))
        raise KeyError(
            f"Unknown ENHSP config {enhsp_config!r}. Known configs: {known}"
        )

    params = ENHSP_CONFIGS[enhsp_config] + f" -timeout {enhsp_timeout}"
    planner = ENHSP(params)

    try:
        plan_res = planner.plan(str(domain_path), str(problem_path))
    except Exception:
        return (
            False,
            "EXCEPTION",
            None,
            traceback.format_exc(),
        )

    planner_status = getattr(plan_res, "status", None)
    planner_status_name = getattr(planner_status, "name", str(planner_status))

    is_success = planner_status == PlanningStatus.SUCCESS

    plan = getattr(plan_res, "plan", None)
    plan_length = len(plan) if plan is not None else None

    detail = f"planner_status={planner_status_name}, plan_length={plan_length}"

    return is_success, planner_status_name, plan_length, detail


def audit_one(
    *,
    ENHSP: Any,
    PlanningStatus: Any,
    ENHSP_CONFIGS: dict[str, str],
    problem_generator_root: Path,
    domain_root: Path,
    domain: str,
    difficulty: str,
    problem_path: Path,
    enhsp_config: str,
    enhsp_timeout: int,
    run_planner: bool,
    solved_index: dict[str, set[str | None]],
    allow_legacy_skip_without_hash: bool,
) -> AuditResult:
    expected_pddl_domain = DOMAINS[domain]["pddl_name"]
    problem_key = make_problem_key(
        domain=domain,
        difficulty=difficulty,
        problem_path=problem_path,
    )

    problem_sha256 = sha256_file(problem_path)
    static_ok, static_detail = static_check_problem(
        problem_path=problem_path,
        expected_pddl_domain=expected_pddl_domain,
    )

    domain_path = find_domain_file(
        problem_generator_root=problem_generator_root,
        domain_root=domain_root,
        domain_name=domain,
    )

    if not static_ok:
        return AuditResult(
            domain=domain,
            difficulty=difficulty,
            problem_key=problem_key,
            problem_sha256=problem_sha256,
            problem_path=problem_path,
            domain_path=domain_path,
            static_ok=False,
            planner_ran=False,
            planner_success=False,
            planner_status="NOT_RUN",
            plan_length=None,
            status="STATIC_FAIL",
            detail=static_detail,
        )

    if not run_planner:
        return AuditResult(
            domain=domain,
            difficulty=difficulty,
            problem_key=problem_key,
            problem_sha256=problem_sha256,
            problem_path=problem_path,
            domain_path=domain_path,
            static_ok=True,
            planner_ran=False,
            planner_success=False,
            planner_status="NOT_RUN",
            plan_length=None,
            status="STATIC_OK",
            detail=static_detail,
        )
    if should_skip_solved_instance(
            problem_key=problem_key,
            problem_sha256=problem_sha256,
            solved_index=solved_index,
            allow_legacy_skip_without_hash=allow_legacy_skip_without_hash,
    ):
        return AuditResult(
            domain=domain,
            difficulty=difficulty,
            problem_key=problem_key,
            problem_sha256=problem_sha256,
            problem_path=problem_path,
            domain_path=domain_path,
            static_ok=True,
            planner_ran=False,
            planner_success=True,
            planner_status="SKIPPED",
            plan_length=None,
            status="SKIPPED_SOLVED",
            detail="skipped because this instance was already solved in a previous audit",
        )

    if domain_path is None:
        return AuditResult(
            domain=domain,
            difficulty=difficulty,
            problem_key=problem_key,
            problem_sha256=problem_sha256,
            problem_path=problem_path,
            domain_path=None,
            static_ok=True,
            planner_ran=False,
            planner_success=False,
            planner_status="NOT_RUN",
            plan_length=None,
            status="NO_DOMAIN_FILE",
            detail="could not locate domain.pddl",
        )

    planner_success, planner_status, plan_length, planner_detail = run_enhsp_planner(
        ENHSP=ENHSP,
        PlanningStatus=PlanningStatus,
        ENHSP_CONFIGS=ENHSP_CONFIGS,
        enhsp_config=enhsp_config,
        enhsp_timeout=enhsp_timeout,
        domain_path=domain_path,
        problem_path=problem_path,
    )

    return AuditResult(
        domain=domain,
        difficulty=difficulty,
        problem_key=problem_key,
        problem_sha256=problem_sha256,
        problem_path=problem_path,
        domain_path=domain_path,
        static_ok=True,
        planner_ran=True,
        planner_success=planner_success,
        planner_status=planner_status,
        plan_length=plan_length,
        status="SOLVED" if planner_success else "PLANNER_FAIL",
        detail=planner_detail,
    )


def write_csv(results: list[AuditResult], output_csv: Path) -> None:
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

        for result in results:
            writer.writerow(
                {
                    "domain": result.domain,
                    "difficulty": result.difficulty,
                    "problem_key": result.problem_key,
                    "problem_sha256": result.problem_sha256,
                    "problem_path": str(result.problem_path),
                    "domain_path": "" if result.domain_path is None else str(result.domain_path),
                    "static_ok": result.static_ok,
                    "planner_ran": result.planner_ran,
                    "planner_success": result.planner_success,
                    "planner_status": result.planner_status,
                    "plan_length": "" if result.plan_length is None else result.plan_length,
                    "status": result.status,
                    "detail": result.detail,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static-check and optionally solve generated validation PDDL instances with ENHSP."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to numeric-asnets project root.",
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
        "--enhsp-config",
        default="hmrmax-astar",
        help="Key in ENHSP_CONFIGS. Default: hmrmax-astar.",
    )

    parser.add_argument(
        "--enhsp-timeout",
        type=int,
        default=15,
        help="ENHSP timeout in seconds per instance. Default: 15.",
    )

    parser.add_argument(
        "--planner",
        action="store_true",
        help="Actually run ENHSP. If omitted, only static checks are performed.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("generated_instances_enhsp_audit.csv"),
        help="CSV output path.",
    )

    parser.add_argument(
        "--domain-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing domain PDDL files. "
            "If omitted, uses --problem-generator-root."
        ),
    )

    parser.add_argument(
        "--use-domain-enhsp-configs",
        action="store_true",
        help=(
            "Use per-domain ENHSP configs from DOMAIN_ENHSP_CONFIGS instead of "
            "the single --enhsp-config value."
        ),
    )

    parser.add_argument(
        "--use-domain-timeouts",
        action="store_true",
        help=(
            "Use per-domain timeouts from DOMAIN_ENHSP_TIMEOUTS instead of "
            "the single --enhsp-timeout value."
        ),
    )

    parser.add_argument(
        "--skip-solved-from",
        type=Path,
        action="append",
        default=[],
        help=(
            "CSV from a previous audit run. Instances with status SOLVED and "
            "matching file hash will be skipped. Can be passed multiple times."
        ),
    )

    parser.add_argument(
        "--solved-statuses",
        default="SOLVED",
        help=(
            "Comma-separated statuses treated as already solved. "
            "Default: SOLVED."
        ),
    )

    parser.add_argument(
        "--allow-legacy-skip-without-hash",
        action="store_true",
        help=(
            "Allow skipping based on domain/difficulty/filename when the previous "
            "CSV has no problem_sha256 column. Useful for old audit CSVs, but less safe "
            "after regenerating files."
        ),
    )

    parser.add_argument(
        "--only-domain",
        default=None,
        help="Optional domain name to audit, e.g. zenotravel.",
    )

    parser.add_argument(
        "--only-difficulty",
        default=None,
        choices=["easy", "medium", "hard"],
        help="Optional difficulty to audit.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    problem_generator_root = args.problem_generator_root.resolve()
    generated_root = args.generated_root.resolve()
    domain_root = (
        args.domain_root.resolve()
        if args.domain_root is not None
        else problem_generator_root
    )

    solved_statuses = {
        status.strip()
        for status in args.solved_statuses.split(",")
        if status.strip()
    }

    solved_index = load_solved_instance_index(
        csv_paths=[path.resolve() for path in args.skip_solved_from],
        solved_statuses=solved_statuses,
    )

    ENHSP, PlanningStatus, ENHSP_CONFIGS = load_enhsp(project_root)

    known_configs = set(ENHSP_CONFIGS.keys())

    if args.enhsp_config not in known_configs:
        known = "\n  ".join(sorted(known_configs))
        raise KeyError(
            f"Unknown --enhsp-config {args.enhsp_config!r}. Known configs:\n  {known}"
        )

    if args.use_domain_enhsp_configs:
        for domain, config_name in DOMAIN_ENHSP_CONFIGS.items():
            if config_name not in known_configs:
                known = "\n  ".join(sorted(known_configs))
                raise KeyError(
                    f"Unknown config for domain {domain!r}: {config_name!r}. "
                    f"Known configs:\n  {known}"
                )

    results = []
    structural_errors = []

    print(f"Project root          : {project_root}")
    print(f"Domain root           : {domain_root}")
    print(f"Problem generator root: {problem_generator_root}")
    print(f"Generated root        : {generated_root}")
    print(f"Expected/difficulty   : {args.expected_per_difficulty}")
    print(f"Planner enabled       : {args.planner}")
    print(f"ENHSP config          : {args.enhsp_config}")
    print(f"ENHSP timeout         : {args.enhsp_timeout}")
    print(f"Use domain configs    : {args.use_domain_enhsp_configs}")
    print(f"Use domain timeouts   : {args.use_domain_timeouts}")
    print(f"ENHSP timeout         : {args.enhsp_timeout}")
    print(f"Skip solved CSVs      : {[str(p) for p in args.skip_solved_from]}")
    print(f"Solved statuses       : {sorted(solved_statuses)}")
    print(f"Previously solved keys: {len(solved_index)}")
    print(f"Legacy skip no hash   : {args.allow_legacy_skip_without_hash}")
    print()

    for domain in DOMAINS:
        if args.only_domain is not None and domain != args.only_domain:
            continue

        for difficulty in DIFFICULTIES:
            if args.only_difficulty is not None and difficulty != args.only_difficulty:
                continue
            folder = generated_root / domain / difficulty

            if not folder.exists():
                structural_errors.append(f"missing folder: {folder}")
                continue

            problems = sorted(folder.glob("*.pddl"))

            if len(problems) != args.expected_per_difficulty:
                structural_errors.append(
                    f"{domain}/{difficulty}: expected "
                    f"{args.expected_per_difficulty}, found {len(problems)}"
                )

            for problem_path in problems:
                domain_enhsp_config = (
                    DOMAIN_ENHSP_CONFIGS.get(domain, args.enhsp_config)
                    if args.use_domain_enhsp_configs
                    else args.enhsp_config
                )

                domain_enhsp_timeout = (
                    DOMAIN_ENHSP_TIMEOUTS.get(domain, args.enhsp_timeout)
                    if args.use_domain_timeouts
                    else args.enhsp_timeout
                )

                result = audit_one(
                    ENHSP=ENHSP,
                    PlanningStatus=PlanningStatus,
                    ENHSP_CONFIGS=ENHSP_CONFIGS,
                    problem_generator_root=problem_generator_root,
                    domain_root=domain_root,
                    domain=domain,
                    difficulty=difficulty,
                    problem_path=problem_path,
                    enhsp_config=domain_enhsp_config,
                    enhsp_timeout=domain_enhsp_timeout,
                    run_planner=args.planner,
                    solved_index=solved_index,
                    allow_legacy_skip_without_hash=args.allow_legacy_skip_without_hash,
                )

                results.append(result)

                print(
                    f"[{result.status:12s}] "
                    f"{domain:15s} {difficulty:6s} "
                    f"{problem_path.name:12s} "
                    f"cfg={domain_enhsp_config:18s} "
                    f"to={domain_enhsp_timeout:<4d} "
                    f"{result.planner_status:15s} "
                    f"len={result.plan_length}"
                )

    write_csv(results, args.output_csv)

    counts = {}

    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print()
    print("===== AUDIT SUMMARY =====")

    for status in sorted(counts):
        print(f"{status:15s}: {counts[status]}")

    if structural_errors:
        print()
        print("===== STRUCTURAL ERRORS =====")
        for error in structural_errors:
            print(error)

    print()
    print(f"Wrote CSV: {args.output_csv}")

    bad_results = [
        result
        for result in results
        if result.status not in {"STATIC_OK", "SOLVED", "SKIPPED_SOLVED"}
    ]

    if structural_errors or bad_results:
        sys.exit(1)


if __name__ == "__main__":
    main()