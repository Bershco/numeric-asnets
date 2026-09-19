import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "materialize_counters_worker_deadline_timeouts_20260919.py"
SPEC = importlib.util.spec_from_file_location("deadline_reconcile", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class DeadlineReconcileTest(unittest.TestCase):
    def test_exact_two_deadline_exits_are_materialized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "complete.jsonl"
            ledger.write_text("".join(
                json.dumps({"instance_number": number, "evaluation_signature": "sig"}) + "\n"
                for number in range(1, 58)
            ))
            log = root / "recovery.txt"
            log.write_text(
                "[EVAL INSTANCE] started number=58 path=/p/58.pddl pid=1\n"
                "[EVAL INSTANCE] started number=59 path=/p/59.pddl pid=2\n"
            )
            summary = module.materialize(
                ledger, log, job_id="7", state="FAILED", exit_code="1:0",
                elapsed=21705,
            )
            self.assertEqual(summary["terminal_after"], 59)
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual([row["instance_number"] for row in records[-2:]], [58, 59])
            self.assertTrue(all(row["status"] == "hard_timeout" for row in records[-2:]))

    def test_short_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "complete.jsonl"
            ledger.write_text("".join(
                json.dumps({"instance_number": number, "evaluation_signature": "sig"}) + "\n"
                for number in range(1, 58)
            ))
            log = root / "recovery.txt"
            log.write_text(
                "[EVAL INSTANCE] started number=58 path=/p/58.pddl pid=1\n"
                "[EVAL INSTANCE] started number=59 path=/p/59.pddl pid=2\n"
            )
            with self.assertRaisesRegex(RuntimeError, "not a six-hour deadline exit"):
                module.materialize(
                    ledger, log, job_id="7", state="FAILED", exit_code="1:0",
                    elapsed=100,
                )


if __name__ == "__main__":
    unittest.main()
