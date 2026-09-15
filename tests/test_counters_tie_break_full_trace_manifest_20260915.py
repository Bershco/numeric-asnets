from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_counters_tie_break_full_trace_manifest_20260915.py"
SPEC = importlib.util.spec_from_file_location("trace_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FullTraceManifestTest(unittest.TestCase):
    def _fixture(self, root: Path) -> list[dict]:
        rows = []
        for task in range(10):
            seed = str(1000 + task)
            policy_log = root / f"policy_{seed}.txt"
            # Exactly two policy successes: evaluator identities 1 and 3.
            policy_log.write_text(
                "[EVAL INSTANCE] completed number=1 status=success steps=1\n"
                "[EVAL INSTANCE] completed number=2 status=finished_unsolved steps=2\n"
                "[EVAL INSTANCE] completed number=3 status=success steps=3\n"
            )
            output = root / f"strict_{seed}"
            output.mkdir()
            ledger = output / f"21233925_{task}.completed.jsonl"
            with ledger.open("w") as stream:
                for number in range(1, 60):
                    # Identity 1 is recovered by action-ID; identity 3 is the
                    # sole wanted policy-success/action-ID-failure identity.
                    status = "finished_unsolved" if number == 3 else "success"
                    stream.write(json.dumps({
                        "instance_number": number,
                        "status": status,
                        "steps": number,
                    }) + "\n")
            rows.append({
                "array_index": str(task),
                "tie_break": "action_id",
                "seed": seed,
                "policy_score": "2",
                "source_training_job_id": f"train-{seed}",
                "source_policy_job_id": f"policy-{seed}",
                "checkpoint": f"/checkpoint/{seed}",
                "source_training_log": f"/train/{seed}.txt",
                "source_policy_log": str(policy_log),
                "remote_output": str(output),
            })
        return rows

    def test_exact_union_is_emitted_twice_without_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, summary = MODULE.build_rows(self._fixture(Path(directory)))
        self.assertEqual(len(rows), 20)
        self.assertEqual({int(row["instance_number"]) for row in rows}, {3})
        self.assertEqual({row["tie_break"] for row in rows}, {"action_id", "policy"})
        self.assertEqual(len({
            (row["tie_break"], row["seed"], row["instance_number"])
            for row in rows
        }), 20)
        self.assertTrue(all(row["policy_success_action_id_failures"] == 1 for row in summary))

    def test_incomplete_source_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._fixture(root)
            ledger = root / "strict_1000" / "21233925_0.completed.jsonl"
            lines = ledger.read_text().splitlines()
            ledger.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                MODULE.build_rows(base)

    def test_duplicate_source_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._fixture(root)
            ledger = root / "strict_1000" / "21233925_0.completed.jsonl"
            first = ledger.read_text().splitlines()[0]
            with ledger.open("a") as stream:
                stream.write(first + "\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                MODULE.build_rows(base)


if __name__ == "__main__":
    unittest.main()
