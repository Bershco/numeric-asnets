from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiment_tracking import materialize_live_stage2_policy as materialize
from experiment_tracking import submit_stage2_policy as submitter


class LiveMaterializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        self.ledger = self.root / "training.tsv"
        self.log = self.root / "training.log"
        self.snapshots = self.root / "snapshots"
        self.output = self.root / "ready.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["value_head", "seed", "teacher"])
            writer.writeheader(); writer.writerow({"value_head": "off", "seed": "7", "teacher": "hmrp-ha-gbfs"})
        with self.ledger.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["value_head", "seed", "slurm_job_id"], delimiter="\t")
            writer.writeheader(); writer.writerow({"value_head": "off", "seed": "7", "slurm_job_id": "123"})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def checkpoint(self, epoch: int, *, stable: bool = True) -> str:
        name = f"snapshot_{epoch}_0.5000"
        path = self.snapshots / name
        path.mkdir(parents=True)
        weights = path / "weights.joblib"
        weights.write_bytes(f"weights-{epoch}".encode())
        if stable:
            old = os.path.getmtime(weights) - 120
            os.utime(weights, (old, old))
        return name

    def run_main(self, state: str) -> list[dict[str, str]]:
        args = [
            "materialize_live_stage2_policy.py", "--training-manifest", str(self.manifest),
            "--training-ledger", str(self.ledger), "--output", str(self.output),
            "--role-prefix", "mprime_final_validation_stage2",
        ]
        with mock.patch.object(sys, "argv", args), mock.patch.object(
            materialize, "accounting", return_value={"123": (state, self.log)}
        ):
            materialize.main()
        with self.output.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_running_emits_stable_every_five_only_and_is_deterministic(self) -> None:
        names = [self.checkpoint(0), self.checkpoint(5), self.checkpoint(6), self.checkpoint(10, stable=False)]
        self.log.write_text(
            f"Snapshot directory: {self.snapshots}\n" +
            "".join(f"[CHECKPOINT SAVED] {name} | ok\n" for name in names), encoding="utf-8"
        )
        first = self.run_main("RUNNING")
        second = self.run_main("RUNNING")
        self.assertEqual(first, second)
        self.assertEqual([row["snapshot_epoch"] for row in first], ["0", "5"])
        self.assertTrue(all(row["analysis_roles"].endswith("learning_curve") for row in first))

    def test_terminal_adds_selected_and_final_roles(self) -> None:
        names = [self.checkpoint(0), self.checkpoint(5), self.checkpoint(99)]
        self.log.write_text(
            f"Snapshot directory: {self.snapshots}\n"
            "[VALIDATION] New best! succ=.5 iter_num=5 snapshot_name=snapshot_5_0.5000\n" +
            "".join(f"[CHECKPOINT SAVED] {name} | ok\n" for name in names), encoding="utf-8"
        )
        rows = self.run_main("COMPLETED")
        self.assertEqual([row["snapshot_epoch"] for row in rows], ["0", "5", "99"])
        roles = {row["snapshot_epoch"]: row["analysis_roles"] for row in rows}
        self.assertIn("validation_selected_policy", roles["5"])
        self.assertIn("final_policy", roles["99"])

    def test_timeout_before_final_epoch_never_assigns_endpoint_roles(self) -> None:
        names = [self.checkpoint(0), self.checkpoint(5), self.checkpoint(50)]
        self.log.write_text(
            f"Snapshot directory: {self.snapshots}\n"
            "[VALIDATION] New best! succ=.5 iter_num=5 snapshot_name=snapshot_5_0.5000\n" +
            "".join(f"[CHECKPOINT SAVED] {name} | ok\n" for name in names), encoding="utf-8"
        )
        rows = self.run_main("TIMEOUT")
        self.assertEqual([row["snapshot_epoch"] for row in rows], ["0", "5", "50"])
        self.assertTrue(all("selected_policy" not in row["analysis_roles"] for row in rows))
        self.assertTrue(all("final_policy" not in row["analysis_roles"] for row in rows))

    def test_terminal_without_best_fails_closed(self) -> None:
        name = self.checkpoint(99)
        self.log.write_text(
            f"Snapshot directory: {self.snapshots}\n[CHECKPOINT SAVED] {name} | ok\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "validation-best"):
            self.run_main("COMPLETED")


class SubmissionDedupeTests(unittest.TestCase):
    def test_second_run_skips_manifest_identity_already_in_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"; checkpoint.mkdir()
            manifest = root / "ready.csv"; ledger = root / "submissions.tsv"
            fields = [
                "manifest_id", "task_type", "domain", "value_head", "seed", "status",
                "source_training_job_id", "snapshot_epoch", "source_checkpoint_ref",
            ]
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                writer.writerow({
                    "manifest_id": "row-1", "task_type": "policy_eval", "domain": "mprime",
                    "value_head": "off", "seed": "7", "status": "ready",
                    "source_training_job_id": "123", "snapshot_epoch": "5",
                    "source_checkpoint_ref": str(checkpoint),
                })
            argv = [
                "submit_stage2_policy.py", "--manifest", str(manifest), "--ledger", str(ledger),
                "--suffix-prefix", "TEST", "--output-prefix", "test", "--one-cycle",
            ]
            calls: list[bool] = []
            def fake_submit(row, dry, suffix_prefix, output_prefix):
                calls.append(dry); return "DRY" if dry else "999"
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                submitter, "submit", side_effect=fake_submit
            ), mock.patch.object(submitter.subprocess, "check_output", return_value=""):
                submitter.main(); submitter.main()
            self.assertEqual(calls, [True, False, True])


if __name__ == "__main__":
    unittest.main()
