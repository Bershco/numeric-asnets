from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_counters_tie_break_full_trace_manifest_20260915.py"
)
SPEC = importlib.util.spec_from_file_location("trace_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FullTraceManifestTests(unittest.TestCase):
    def make_rows(self, root: Path) -> list[dict[str, str]]:
        rows = []
        for task in range(10):
            seed = str(1000 + task)
            output = root / seed
            output.mkdir()
            policy_log = root / f"policy_{seed}.txt"
            # fz_instance_2 is evaluator identity 1.
            policy_log.write_text(
                "[EVAL][PLAN] instance=../problems/fz_instance_2.pddl\n",
                encoding="utf-8",
            )
            ledger = output / f"{MODULE.SOURCE_ARRAY_JOB_ID}_{task}.completed.jsonl"
            records = []
            for number in range(1, MODULE.TOTAL_INSTANCES + 1):
                status = "finished_unsolved" if task == 0 and number == 1 else "success"
                records.append(json.dumps({
                    "instance_number": number,
                    "status": status,
                    "steps": 10000 if status != "success" else number,
                }))
            ledger.write_text("\n".join(records) + "\n", encoding="utf-8")
            rows.append({
                "array_index": str(task),
                "tie_break": "action_id",
                "seed": seed,
                "policy_score": "1",
                "source_policy_log": str(policy_log),
                "remote_output": str(output),
                "source_training_job_id": f"train-{task}",
                "source_policy_job_id": f"policy-{task}",
                "checkpoint": f"checkpoint-{task}",
                "source_training_log": f"training-{task}.txt",
            })
        return rows

    def test_emits_only_policy_success_action_id_failure_under_two_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trace, summary = MODULE.build_rows(self.make_rows(Path(temp)))
        self.assertEqual(len(trace), 2)
        self.assertEqual({row["tie_break"] for row in trace}, {"action_id", "policy"})
        self.assertEqual({row["instance_number"] for row in trace}, {1})
        self.assertEqual(sum(int(row["policy_success_action_id_failures"])
                             for row in summary), 1)

    def test_rejects_an_incomplete_action_id_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            rows = self.make_rows(Path(temp))
            ledger = Path(rows[0]["remote_output"]) / (
                f"{MODULE.SOURCE_ARRAY_JOB_ID}_0.completed.jsonl"
            )
            ledger.write_text("\n".join(ledger.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                MODULE.build_rows(rows)


if __name__ == "__main__":
    unittest.main()
