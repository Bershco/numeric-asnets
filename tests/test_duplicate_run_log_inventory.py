import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.inventory_duplicate_run_logs import (
    prepare_from_paths,
    scan_shard,
    summarize,
)


class DuplicateRunLogInventoryTests(unittest.TestCase):
    def _run_pair(self, root: Path, command: bytes, prefix: str, exact: bool = True):
        digest = hashlib.md5(command).hexdigest()
        run_dir = root / "runs" / digest
        info_dir = root / prefix / "run-info"
        run_dir.mkdir(parents=True)
        info_dir.mkdir(parents=True)
        payload = {
            "cmdline": command,
            "stdout": f"Unique prefix: {prefix}\nresult\n".encode(),
            "stderr": b"",
            "elapsed_secs": b"1.0\n",
            "termination_status": b"timed_out: False\nbad_retcode: False\n",
        }
        for name, content in payload.items():
            (run_dir / name).write_bytes(content)
            (info_dir / name).write_bytes(content)
        if not exact:
            (info_dir / "elapsed_secs").write_bytes(b"2.0\n")
        return digest, run_dir, info_dir

    def test_exact_mismatch_missing_and_orphan_are_distinguished(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment-results"
            campaign = Path(temporary) / "campaign"
            exact_digest, exact_run, exact_info = self._run_pair(root, b"python exact", "exact")
            _, mismatch_run, mismatch_info = self._run_pair(
                root, b"python mismatch", "mismatch", exact=False
            )
            missing_info = root / "missing" / "run-info"
            missing_info.mkdir(parents=True)
            (missing_info / "cmdline").write_bytes(b"python missing")
            orphan = root / "runs" / hashlib.md5(b"python orphan").hexdigest()
            orphan.mkdir(parents=True)
            (orphan / "cmdline").write_bytes(b"python orphan")
            (orphan / "stdout").write_bytes(b"failed before Unique prefix\n")
            (orphan / "stderr").write_bytes(b"boom\n")

            metadata = prepare_from_paths(
                root,
                campaign,
                2,
                [exact_info, mismatch_info, missing_info],
                [exact_run, mismatch_run, orphan],
            )
            self.assertEqual(metadata["tasks"], 4)
            for shard in range(2):
                scan_shard(SimpleNamespace(campaign=campaign, shard=shard))
            summarize(SimpleNamespace(campaign=campaign))

            summary = json.loads((campaign / "summary.json").read_text())
            self.assertTrue(summary["campaign_complete"])
            self.assertEqual(summary["eligible_unique_run_directories"], 1)
            self.assertEqual(summary["status_counts"]["exact_duplicate_tree"], 1)
            self.assertEqual(
                summary["status_counts"]["log_triplet_exact_but_tree_not_exact"], 1
            )
            self.assertEqual(summary["status_counts"]["missing_run_dir"], 1)
            self.assertEqual(summary["status_counts"]["orphan_run_no_run_info"], 1)

            candidates = (campaign / "deletion_candidates.csv").read_text()
            self.assertIn(exact_digest, candidates)
            self.assertNotIn(mismatch_run.name, candidates)
            self.assertTrue((campaign / "exceptions.jsonl").is_file())

    def test_resume_skips_already_durable_result(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment-results"
            campaign = Path(temporary) / "campaign"
            _, run_dir, info_dir = self._run_pair(root, b"python resume", "resume")
            prepare_from_paths(root, campaign, 1, [info_dir], [run_dir])
            scan_shard(SimpleNamespace(campaign=campaign, shard=0))
            done = campaign / "results" / "shard_0000.done"
            done.unlink()
            before = (campaign / "results" / "shard_0000.jsonl").read_text()
            scan_shard(SimpleNamespace(campaign=campaign, shard=0))
            after = (campaign / "results" / "shard_0000.jsonl").read_text()
            self.assertEqual(before, after)

    def test_same_command_under_two_experiment_prefixes_is_not_collapsed(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment-results"
            campaign = Path(temporary) / "campaign"
            command = b"python shared-command"
            pairs = []
            for prefix_name in ("experiment-a", "experiment-b"):
                prefix_root = root / prefix_name
                digest, run_dir, info_dir = self._run_pair(
                    prefix_root, command, f"output-{prefix_name}"
                )
                pairs.append((digest, run_dir, info_dir))
            metadata = prepare_from_paths(
                root,
                campaign,
                1,
                [pair[2] for pair in pairs],
                [pair[1] for pair in pairs],
            )
            self.assertEqual(metadata["run_trees_referenced_by_run_info"], 2)
            self.assertEqual(metadata["tasks"], 2)
            scan_shard(SimpleNamespace(campaign=campaign, shard=0))
            summarize(SimpleNamespace(campaign=campaign))
            summary = json.loads((campaign / "summary.json").read_text())
            self.assertEqual(summary["eligible_unique_run_directories"], 2)


if __name__ == "__main__":
    unittest.main()
