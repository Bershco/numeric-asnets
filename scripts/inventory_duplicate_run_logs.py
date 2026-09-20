#!/usr/bin/env python3
"""Inventory duplicated ASNet launcher logs without deleting anything.

The legacy launchers write ``runs/<md5(command)>`` and, after a successful
run, copy that directory to ``<scientific-output>/run-info``.  This program
builds a sharded inventory of those pairs, byte-compares the complete trees,
and reports the logical and allocated bytes that *could* be reclaimed by a
later, separately approved deletion campaign.

Discovery deliberately delegates the NFS directory walk to ``find``.  Python
only consumes its NUL-delimited output and works on bounded shard manifests;
it never performs a monolithic ``Path.rglob`` over the experiment tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Iterator


LOG_FILES = ("cmdline", "stdout", "stderr")
CHUNK_SIZE = 8 * 1024 * 1024


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _iter_find_nul(command: list[str]) -> Iterator[Path]:
    """Yield paths from a NUL-delimited GNU find invocation."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    pending = b""
    while True:
        chunk = proc.stdout.read(1024 * 1024)
        if not chunk:
            break
        pending += chunk
        fields = pending.split(b"\0")
        pending = fields.pop()
        for field in fields:
            if field:
                yield Path(os.fsdecode(field))
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    returncode = proc.wait()
    if returncode:
        raise RuntimeError(f"find failed ({returncode}): {stderr.strip()}")
    if pending:
        raise RuntimeError("find emitted a non-NUL-terminated path")


def discover_layout(root: Path) -> tuple[list[Path], list[Path]]:
    """Discover run-info and run trees in one NUL-safe GNU find traversal."""
    found = _iter_find_nul(
        [
            "find",
            os.fspath(root),
            "-type",
            "d",
            "(",
            "-name",
            "run-info",
            "-o",
            "-name",
            "runs",
            ")",
            "-print0",
        ]
    )
    run_info_dirs: list[Path] = []
    run_dirs: list[Path] = []
    for path in found:
        if path.name == "run-info":
            run_info_dirs.append(path)
            continue
        if path.name != "runs":
            continue
        try:
            with os.scandir(path) as listing:
                for entry in listing:
                    if entry.is_dir(follow_symlinks=False):
                        run_dirs.append(Path(entry.path))
        except OSError:
            # Missing/unreadable counterparts are recorded from run-info.
            continue
    return run_info_dirs, run_dirs


def command_digest(cmdline_path: Path) -> str:
    return hashlib.md5(cmdline_path.read_bytes()).hexdigest()


def infer_run_dir(root: Path, info_dir: Path, digest: str) -> Path:
    """Resolve the nearest experiment-prefix ``runs/<digest>`` partner."""
    resolved_root = root.resolve()
    current = info_dir.parent
    ancestors: list[Path] = []
    while True:
        ancestors.append(current)
        if current == resolved_root or current.parent == current:
            break
        current = current.parent
    for ancestor in ancestors:
        candidate = ancestor / "runs" / digest
        if candidate.is_dir():
            return candidate
    for ancestor in ancestors:
        runs = ancestor / "runs"
        if runs.is_dir():
            return runs / digest
    return resolved_root / "runs" / digest


def prepare_from_paths(
    root: Path,
    campaign: Path,
    shard_count: int,
    run_info_dirs: Iterable[Path],
    run_dirs: Iterable[Path],
) -> dict:
    """Create stable JSONL shard manifests from already discovered paths."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    campaign.mkdir(parents=True, exist_ok=True)
    shard_dir = campaign / "shards"
    shard_dir.mkdir(exist_ok=True)
    error_path = campaign / "prepare_exceptions.jsonl"

    candidates: dict[str, dict] = {}
    exceptions: list[dict] = []
    run_info_count = 0
    for info_dir in run_info_dirs:
        run_info_count += 1
        cmdline = info_dir / "cmdline"
        if not cmdline.is_file():
            exceptions.append(
                {"status": "run_info_missing_cmdline", "run_info_dir": os.fspath(info_dir)}
            )
            continue
        try:
            digest = command_digest(cmdline)
        except OSError as exc:
            exceptions.append(
                {
                    "status": "run_info_cmdline_unreadable",
                    "run_info_dir": os.fspath(info_dir),
                    "error": repr(exc),
                }
            )
            continue
        inferred_run = infer_run_dir(root, info_dir.resolve(), digest)
        run_key = os.path.normcase(os.path.normpath(os.fspath(inferred_run)))
        entry = candidates.setdefault(
            run_key,
            {"digest": digest, "run_dir": os.fspath(inferred_run), "run_info_dirs": []},
        )
        entry["run_info_dirs"].append(os.fspath(info_dir))

    discovered_runs: dict[str, str] = {}
    for run_dir in run_dirs:
        run_key = os.path.normcase(os.path.normpath(os.fspath(run_dir.resolve())))
        discovered_runs[run_key] = os.fspath(run_dir.resolve())

    all_run_paths = sorted(set(candidates) | set(discovered_runs))
    handles = [
        (shard_dir / f"shard_{index:04d}.jsonl").open("w", encoding="utf-8", newline="\n")
        for index in range(shard_count)
    ]
    shard_counts = [0] * shard_count
    try:
        for run_path_key in all_run_paths:
            candidate = candidates.get(run_path_key)
            run_path = discovered_runs.get(
                run_path_key,
                candidate["run_dir"] if candidate else run_path_key,
            )
            digest = candidate["digest"] if candidate else Path(run_path).name
            key = hashlib.sha256(run_path_key.encode("utf-8", errors="surrogatepass")).hexdigest()
            task = {
                "key": key,
                "digest": digest,
                "run_dir": run_path,
                "run_info_dirs": sorted(candidate["run_info_dirs"] if candidate else []),
            }
            shard = int(key[:8], 16) % shard_count
            handles[shard].write(json.dumps(task, sort_keys=True) + "\n")
            shard_counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()

    with error_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in exceptions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    metadata = {
        "schema_version": 1,
        "created_at": _now(),
        "root": os.fspath(root.resolve()),
        "shard_count": shard_count,
        "run_info_directories": run_info_count,
        "run_trees_referenced_by_run_info": len(candidates),
        "run_directories": len(discovered_runs),
        "tasks": len(all_run_paths),
        "prepare_exceptions": len(exceptions),
        "shard_task_counts": shard_counts,
        "deletion_performed": False,
    }
    (campaign / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (campaign / ".prepare.complete").write_text(_now() + "\n", encoding="utf-8")
    return metadata


def prepare(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    campaign = args.campaign.resolve()
    if (campaign / ".prepare.complete").exists():
        print((campaign / "metadata.json").read_text(encoding="utf-8"), end="")
        return
    if campaign.exists() and any(campaign.iterdir()):
        raise RuntimeError(
            f"incomplete/nonempty campaign directory {campaign}; preserve it and use a new directory"
        )
    run_info_dirs, run_dirs = discover_layout(root)
    metadata = prepare_from_paths(
        root,
        campaign,
        args.shards,
        run_info_dirs,
        run_dirs,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _allocated(stat_result: os.stat_result) -> int:
    return int(getattr(stat_result, "st_blocks", 0)) * 512


def tree_inventory(root: Path) -> tuple[dict[str, dict], dict]:
    """Inventory a small run-info tree without following links."""
    entries: dict[str, dict] = {}
    totals = {"logical_bytes": 0, "allocated_bytes": 0, "files": 0, "directories": 0}
    stack = [root]
    while stack:
        current = stack.pop()
        stat_result = current.lstat()
        rel = "." if current == root else current.relative_to(root).as_posix()
        if current.is_symlink():
            entries[rel] = {"type": "symlink", "target": os.readlink(current)}
            totals["allocated_bytes"] += _allocated(stat_result)
            continue
        if current.is_dir():
            entries[rel] = {"type": "directory"}
            totals["directories"] += 1
            totals["allocated_bytes"] += _allocated(stat_result)
            with os.scandir(current) as listing:
                children = [Path(entry.path) for entry in listing]
            stack.extend(children)
            continue
        if current.is_file():
            entries[rel] = {
                "type": "file",
                "size": stat_result.st_size,
                "link_count": stat_result.st_nlink,
            }
            totals["files"] += 1
            totals["logical_bytes"] += stat_result.st_size
            totals["allocated_bytes"] += _allocated(stat_result)
            continue
        entries[rel] = {"type": "unsupported", "mode": stat_result.st_mode}
        totals["allocated_bytes"] += _allocated(stat_result)
    return entries, totals


def compare_file_pair(left: Path, right: Path) -> tuple[bool, str, str]:
    left_hash = hashlib.sha256()
    right_hash = hashlib.sha256()
    equal = True
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(CHUNK_SIZE)
            right_chunk = right_handle.read(CHUNK_SIZE)
            if not left_chunk and not right_chunk:
                break
            left_hash.update(left_chunk)
            right_hash.update(right_chunk)
            if left_chunk != right_chunk:
                equal = False
    return equal, left_hash.hexdigest(), right_hash.hexdigest()


def compare_trees(run_dir: Path, run_info: Path, run_entries: dict[str, dict]) -> dict:
    try:
        info_entries, _ = tree_inventory(run_info)
    except OSError as exc:
        return {"run_info_dir": os.fspath(run_info), "status": "run_info_unreadable", "error": repr(exc)}
    run_keys = set(run_entries)
    info_keys = set(info_entries)
    if run_keys != info_keys:
        return {
            "run_info_dir": os.fspath(run_info),
            "status": "tree_entry_set_mismatch",
            "only_in_run": sorted(run_keys - info_keys),
            "only_in_run_info": sorted(info_keys - run_keys),
        }
    metadata_mismatches = [
        rel for rel in sorted(run_keys) if run_entries[rel] != info_entries[rel]
    ]
    if metadata_mismatches:
        return {
            "run_info_dir": os.fspath(run_info),
            "status": "tree_metadata_mismatch",
            "mismatched_entries": metadata_mismatches,
        }

    hashes: dict[str, dict[str, str]] = {}
    mismatched_files: list[str] = []
    for rel in sorted(run_keys):
        if run_entries[rel]["type"] != "file":
            continue
        equal, left_hash, right_hash = compare_file_pair(run_dir / rel, run_info / rel)
        hashes[rel] = {"run_sha256": left_hash, "run_info_sha256": right_hash}
        if not equal:
            mismatched_files.append(rel)
    return {
        "run_info_dir": os.fspath(run_info),
        "status": "exact_tree_match" if not mismatched_files else "tree_content_mismatch",
        "mismatched_files": mismatched_files,
        "log_triplet_exact": all(
            rel in hashes and hashes[rel]["run_sha256"] == hashes[rel]["run_info_sha256"]
            for rel in LOG_FILES
        ),
        "file_sha256": hashes,
    }


def _extract_unique_prefix(stdout_path: Path, limit: int = 32 * 1024 * 1024) -> str | None:
    read = 0
    try:
        with stdout_path.open("rb") as handle:
            for line in handle:
                read += len(line)
                if line.startswith(b"Unique prefix: "):
                    return os.fsdecode(line[len(b"Unique prefix: ") :].rstrip(b"\r\n"))
                if read >= limit:
                    break
    except OSError:
        return None
    return None


def scan_task(task: dict) -> dict:
    started = _now()
    digest = task["digest"]
    run_dir = Path(task["run_dir"])
    info_dirs = [Path(item) for item in task["run_info_dirs"]]
    base = {
        "key": task["key"],
        "digest": digest,
        "run_dir": os.fspath(run_dir),
        "candidate_count": len(info_dirs),
        "scan_started_at": started,
        "deletion_performed": False,
    }
    if not run_dir.is_dir():
        return {**base, "status": "missing_run_dir", "eligible": False, "scan_finished_at": _now()}
    try:
        run_entries, totals = tree_inventory(run_dir)
    except OSError as exc:
        return {
            **base,
            "status": "run_dir_unreadable",
            "eligible": False,
            "error": repr(exc),
            "scan_finished_at": _now(),
        }
    base.update({f"run_{key}": value for key, value in totals.items()})
    safe_entry_types = all(
        entry["type"] in {"file", "directory"} for entry in run_entries.values()
    )
    base["safe_entry_types"] = safe_entry_types
    singly_linked_files = all(
        entry.get("link_count", 1) == 1
        for entry in run_entries.values()
        if entry["type"] == "file"
    )
    base["singly_linked_files"] = singly_linked_files
    cmdline = run_dir / "cmdline"
    actual_digest = None
    if cmdline.is_file():
        try:
            actual_digest = command_digest(cmdline)
        except OSError:
            pass
    base["run_cmdline_digest"] = actual_digest
    base["digest_matches_run_cmdline"] = actual_digest == digest
    base["unique_prefix"] = _extract_unique_prefix(run_dir / "stdout")

    termination = run_dir / "termination_status"
    if termination.is_file():
        try:
            base["termination_status"] = termination.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError:
            base["termination_status"] = None

    if not info_dirs:
        return {
            **base,
            "status": "orphan_run_no_run_info",
            "eligible": False,
            "comparisons": [],
            "scan_finished_at": _now(),
        }

    comparisons = [compare_trees(run_dir, info_dir, run_entries) for info_dir in info_dirs]
    exact = [row for row in comparisons if row["status"] == "exact_tree_match"]
    prefix = base["unique_prefix"]
    expected_info = None
    if prefix:
        expected_info = os.path.normpath(os.fspath(run_dir.parent.parent / prefix / "run-info"))
    prefix_matched = expected_info is None or any(
        os.path.normpath(row["run_info_dir"]) == expected_info for row in exact
    )
    base["stdout_prefix_expected_run_info"] = expected_info
    base["stdout_prefix_matches_exact_candidate"] = prefix_matched
    eligible = (
        bool(exact)
        and actual_digest == digest
        and safe_entry_types
        and singly_linked_files
        and prefix_matched
    )
    if eligible:
        status = "exact_duplicate_tree"
    elif exact:
        status = "exact_tree_but_digest_mismatch"
    elif any(row.get("log_triplet_exact") for row in comparisons):
        status = "log_triplet_exact_but_tree_not_exact"
    else:
        status = "no_exact_duplicate"
    return {
        **base,
        "status": status,
        "eligible": eligible,
        "retained_run_info_dirs": [row["run_info_dir"] for row in exact],
        "comparisons": comparisons,
        "scan_finished_at": _now(),
    }


def _valid_completed_keys(result_path: Path) -> set[str]:
    completed: set[str] = set()
    if not result_path.exists():
        return completed
    with result_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("key"):
                completed.add(str(row["key"]))
    return completed


def scan_shard(args: argparse.Namespace) -> None:
    campaign = args.campaign.resolve()
    shard_path = campaign / "shards" / f"shard_{args.shard:04d}.jsonl"
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    result_dir = campaign / "results"
    result_dir.mkdir(exist_ok=True)
    result_path = result_dir / f"shard_{args.shard:04d}.jsonl"
    done_path = result_dir / f"shard_{args.shard:04d}.done"
    if done_path.exists():
        print(f"shard {args.shard} already complete")
        return
    completed = _valid_completed_keys(result_path)
    task_count = 0
    scanned = 0
    with shard_path.open("r", encoding="utf-8") as source, result_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            task_count += 1
            task = json.loads(line)
            if task["key"] in completed:
                continue
            try:
                result = scan_task(task)
            except Exception as exc:  # fail closed and keep the shard resumable
                result = {
                    "key": task.get("key"),
                    "digest": task.get("digest"),
                    "run_dir": task.get("run_dir"),
                    "status": "scan_exception",
                    "eligible": False,
                    "error": repr(exc),
                    "scan_finished_at": _now(),
                    "deletion_performed": False,
                }
            destination.write(json.dumps(result, sort_keys=True) + "\n")
            destination.flush()
            scanned += 1
            if scanned % getattr(args, "fsync_every", 25) == 0:
                os.fsync(destination.fileno())
        destination.flush()
        os.fsync(destination.fileno())
    final_keys = _valid_completed_keys(result_path)
    if len(final_keys) != task_count:
        raise RuntimeError(
            f"shard {args.shard} incomplete: {len(final_keys)}/{task_count} unique results"
        )
    done_path.write_text(
        json.dumps({"completed_at": _now(), "tasks": task_count, "newly_scanned": scanned}) + "\n",
        encoding="utf-8",
    )
    print(f"shard {args.shard}: {task_count}/{task_count} complete ({scanned} new)")


def summarize(args: argparse.Namespace) -> None:
    campaign = args.campaign.resolve()
    metadata = json.loads((campaign / "metadata.json").read_text(encoding="utf-8"))
    rows: dict[str, dict] = {}
    for result_path in sorted((campaign / "results").glob("shard_*.jsonl")):
        with result_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("key"):
                    rows[str(row["key"])] = row
    status_counts: dict[str, int] = {}
    logical = allocated = eligible = 0
    for row in rows.values():
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if row.get("eligible"):
            eligible += 1
            logical += int(row.get("run_logical_bytes", 0))
            allocated += int(row.get("run_allocated_bytes", 0))

    done_shards = len(list((campaign / "results").glob("shard_*.done")))
    summary = {
        "schema_version": 1,
        "generated_at": _now(),
        "root": metadata["root"],
        "expected_tasks": metadata["tasks"],
        "result_tasks": len(rows),
        "expected_shards": metadata["shard_count"],
        "complete_shards": done_shards,
        "campaign_complete": len(rows) == metadata["tasks"] and done_shards == metadata["shard_count"],
        "eligible_unique_run_directories": eligible,
        "reclaimable_logical_bytes": logical,
        "reclaimable_allocated_bytes": allocated,
        "status_counts": dict(sorted(status_counts.items())),
        "deletion_performed": False,
    }
    (campaign / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    candidate_fields = [
        "digest",
        "run_dir",
        "retained_run_info_dir",
        "run_logical_bytes",
        "run_allocated_bytes",
        "run_files",
        "run_directories",
        "scan_finished_at",
    ]
    with (campaign / "deletion_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda item: str(item.get("digest", ""))):
            if not row.get("eligible"):
                continue
            retained = row.get("retained_run_info_dirs") or []
            writer.writerow(
                {
                    "digest": row.get("digest"),
                    "run_dir": row.get("run_dir"),
                    "retained_run_info_dir": retained[0] if retained else "",
                    "run_logical_bytes": row.get("run_logical_bytes", 0),
                    "run_allocated_bytes": row.get("run_allocated_bytes", 0),
                    "run_files": row.get("run_files", 0),
                    "run_directories": row.get("run_directories", 0),
                    "scan_finished_at": row.get("scan_finished_at", ""),
                }
            )
    with (campaign / "exceptions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows.values(), key=lambda item: str(item.get("key", ""))):
            if not row.get("eligible"):
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    gib = 1024 ** 3
    lines = [
        "# Duplicate launcher-log inventory",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "This is an inventory-only result. **No file was deleted.** A future deletion",
        "must revalidate every candidate immediately before removal.",
        "",
        f"- Tasks: {summary['result_tasks']}/{summary['expected_tasks']}",
        f"- Shards: {summary['complete_shards']}/{summary['expected_shards']}",
        f"- Complete: {summary['campaign_complete']}",
        f"- Fully byte-identical `runs/<digest>` trees: {eligible}",
        f"- Reclaimable logical bytes: {logical:,} ({logical / gib:.3f} GiB)",
        f"- Reclaimable allocated bytes: {allocated:,} ({allocated / gib:.3f} GiB)",
        "",
        "## Status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{status}` | {count} |" for status, count in summary["status_counts"].items())
    lines.extend(
        [
            "",
            "`deletion_candidates.csv` lists only full-tree, digest-verified matches.",
            "`exceptions.jsonl` retains every mismatch, missing partner, orphan or scan error.",
            "",
        ]
    )
    (campaign / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--campaign", type=Path, required=True)
    prepare_parser.add_argument("--shards", type=int, default=32)
    prepare_parser.set_defaults(func=prepare)

    scan_parser = subparsers.add_parser("scan-shard")
    scan_parser.add_argument("--campaign", type=Path, required=True)
    scan_parser.add_argument("--shard", type=int, required=True)
    scan_parser.add_argument("--fsync-every", type=int, default=25)
    scan_parser.set_defaults(func=scan_shard)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--campaign", type=Path, required=True)
    summarize_parser.set_defaults(func=summarize)
    return result


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
