from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reconcile_mprime_final_stage2_search_20260917 import (
    evidence_from_ledger,
    evidence_from_log,
    identity_paths,
    merge_evidence,
)


class MPrimeStage2ReconciliationTest(unittest.TestCase):
    def test_ledger_success_and_action_limit_are_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed.jsonl"
            records = [
                {
                    "instance_number": 1,
                    "instance_path": "../instances/pfile01.pddl",
                    "status": "success", "hit_goal": True, "steps": 5,
                },
                {
                    "instance_number": 2,
                    "instance_path": "../instances/pfile02.pddl",
                    "status": "finished_unsolved", "hit_goal": False,
                    "steps": 10000,
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            evidence = evidence_from_ledger(path, 10000)
            self.assertEqual(evidence[1][0].classification, "success")
            self.assertEqual(evidence[2][0].classification, "action_limit")

    def test_only_declared_six_hour_timeout_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.txt"
            path.write_text(
                "[EVAL INSTANCE] timeout number=3 "
                "path=../instances/pfile04.pddl limit=21600.0s\n"
                "[EVAL INSTANCE] timeout number=4 "
                "path=../instances/pfile05.pddl limit=60.0s\n",
                encoding="utf-8",
            )
            evidence = evidence_from_log(
                path, declared_timeout=21600.0, max_actions=10000
            )
            self.assertEqual(evidence[3][0].classification, "six_hour_timeout")
            self.assertNotIn(4, evidence)

    def test_instance_scoped_oom_is_recovery_hint_but_job_level_oom_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.txt"
            path.write_text(
                "[EVAL INSTANCE] crashed number=5 "
                "path=../instances/pfile09.pddl elapsed=10 error=Traceback\n"
                "java.lang.OutOfMemoryError: Java heap space\n"
                "[EVAL INSTANCE] started number=6 path=../instances/pfile13.pddl pid=1\n"
                "slurmstepd: error: Detected 1 oom-kill event(s) in StepId=1.batch\n",
                encoding="utf-8",
            )
            evidence = evidence_from_log(
                path, declared_timeout=21600.0, max_actions=10000
            )
            self.assertEqual(evidence[5][0].classification, "instance_scoped_oom")
            self.assertNotIn(6, evidence)
            terminal, conflicts, oom_hints = merge_evidence({}, [evidence])
            self.assertNotIn(5, terminal)
            self.assertEqual(conflicts, {})
            self.assertEqual(
                oom_hints[5][0].classification,
                "instance_scoped_oom",
            )

    def test_conflicting_terminal_records_require_manual_review(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "completed.jsonl"
            ledger_path.write_text(json.dumps({
                "instance_number": 1,
                "instance_path": "../instances/pfile01.pddl",
                "status": "success", "hit_goal": True, "steps": 5,
            }) + "\n", encoding="utf-8")
            log_path = Path(directory) / "attempt.txt"
            log_path.write_text(
                "[EVAL INSTANCE] timeout number=1 "
                "path=../instances/pfile01.pddl limit=21600.0s\n",
                encoding="utf-8",
            )
            terminal, conflicts, oom_hints = merge_evidence(
                evidence_from_ledger(ledger_path, 10000),
                [evidence_from_log(
                    log_path, declared_timeout=21600.0, max_actions=10000
                )],
            )
            self.assertNotIn(1, terminal)
            self.assertEqual(oom_hints, {})
            self.assertEqual(
                {entry.classification for entry in conflicts[1]},
                {"success", "six_hour_timeout"},
            )

    def test_plain_worker_death_is_not_a_scientific_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.txt"
            path.write_text(
                "[EVAL INSTANCE] died number=7 "
                "path=../instances/pfile14.pddl exitcode=-9\n",
                encoding="utf-8",
            )
            evidence = evidence_from_log(
                path, declared_timeout=21600.0, max_actions=10000
            )
            self.assertEqual(evidence, {})

    def test_recovery_ledgers_remain_separate_and_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "fixed" / "off" / "1"
            original = identity / "completion" / "row.jsonl"
            recovery = identity / "recovery_completion" / "instance_6_job.jsonl"
            original.parent.mkdir(parents=True)
            recovery.parent.mkdir(parents=True)
            original.write_text("", encoding="utf-8")
            recovery.write_text("", encoding="utf-8")
            ledgers, _ = identity_paths(root, {
                "search_method": "fixed", "value_head": "off", "seed": "1",
                "manifest_id": "row", "array_index": "0",
            })
            self.assertEqual(ledgers, [original, recovery])


if __name__ == "__main__":
    unittest.main()
