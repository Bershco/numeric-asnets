from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "reconcile_counters_tie_break_terminal_records_20260915.py"
SPEC = importlib.util.spec_from_file_location("terminal_reconcile", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TerminalReconcileTests(unittest.TestCase):
    def _ledger(self, root: Path, instance_path: str = "instance_1.pddl") -> Path:
        ledger = root / "done.jsonl"
        ledger.write_text(json.dumps({
            "evaluation_signature": "sig",
            "instance_number": 1,
            "instance_path": instance_path,
            "status": "success",
            "hit_goal": True,
            "steps": 3,
            "plan": [],
            "elapsed_seconds": 1.0,
        }) + "\n")
        return ledger

    def test_adds_timeout_but_not_crash_and_durable_success_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = self._ledger(root)
            log = root / "attempt.txt"
            log.write_text(
                "[EVAL INSTANCE] timeout number=1 path=instance_1.pddl limit=21600.0s\n"
                "[EVAL INSTANCE] timeout number=2 path=instance_2.pddl limit=21600.0s\n"
                "[EVAL INSTANCE] crashed number=3 path=instance_3.pddl elapsed=4 error=x\n"
                "[EVAL INSTANCE] started number=4 path=instance_4.pddl pid=1\n"
            )
            result = MODULE.reconcile(ledger, [log])
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        self.assertEqual(result["timeouts_added"], 1)
        self.assertEqual(result["timeout_events_already_durable"], 1)
        self.assertEqual([row["instance_number"] for row in rows], [1, 2])
        self.assertEqual(rows[1]["status"], "hard_timeout")
        self.assertEqual(rows[1]["evaluation_signature"], "sig")
        self.assertEqual(rows[1]["terminal_evidence_line"], 2)

    def test_rejects_path_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = self._ledger(root)
            log = root / "attempt.txt"
            log.write_text("[EVAL INSTANCE] timeout number=1 path=wrong.pddl limit=21600.0s\n")
            with self.assertRaisesRegex(RuntimeError, "path mismatch"):
                MODULE.reconcile(ledger, [log])

    def test_rejects_wrong_timeout_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = self._ledger(root)
            log = root / "attempt.txt"
            log.write_text("[EVAL INSTANCE] timeout number=2 path=instance_2.pddl limit=12.0s\n")
            with self.assertRaisesRegex(RuntimeError, "unexpected timeout cap"):
                MODULE.reconcile(ledger, [log])


if __name__ == "__main__":
    unittest.main()
