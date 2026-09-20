import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_counters_full_trace_recovery_20260920.py"


class CountersTraceRecoveryTests(unittest.TestCase):
    def test_retains_only_valid_ledger_with_steps(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = []
            for index in range(3):
                output = root / f"source_{index}"
                output.mkdir()
                rows.append({
                    "array_index": index,
                    "tie_break": "action_id",
                    "seed": str(100 + index),
                    "instance_number": str(index + 1),
                    "remote_output": str(output),
                })
            valid = Path(rows[0]["remote_output"]) / "valid.completed.jsonl"
            valid.write_text(json.dumps({"instance_number": 1}) + "\n")
            valid.with_name("valid_steps.csv").write_text("step\n")
            no_steps = Path(rows[1]["remote_output"]) / "partial.completed.jsonl"
            no_steps.write_text(json.dumps({"instance_number": 2}) + "\n")
            malformed = Path(rows[2]["remote_output"]) / "bad.completed.jsonl"
            malformed.write_text("not json\n")

            source = root / "source.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "recovery.csv"
            summary = root / "summary.json"
            subprocess.run([
                sys.executable, str(SCRIPT), "--source", str(source),
                "--output", str(output), "--summary", str(summary),
                "--recovery-root", str(root / "recovery"),
            ], check=True)
            with output.open(newline="") as stream:
                recovered = list(csv.DictReader(stream))
            self.assertEqual([int(row["source_trace_index"]) for row in recovered], [1, 2])
            self.assertEqual(json.loads(summary.read_text())["retained_complete"], 1)

    def test_duplicate_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [{
                "array_index": index,
                "tie_break": "policy",
                "seed": "42",
                "instance_number": "7",
                "remote_output": str(root / f"source_{index}"),
            } for index in range(2)]
            source = root / "source.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--source", str(source),
                "--output", str(root / "recovery.csv"),
                "--summary", str(root / "summary.json"),
                "--recovery-root", str(root / "recovery"),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate identities", result.stderr)


if __name__ == "__main__":
    unittest.main()
