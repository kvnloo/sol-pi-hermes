import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hermes_ab_dry_run.py"
ARMS = ("base", "observation_pack", "action_fusion", "both")


class HermesABDryRunTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *args],
                              text=True, capture_output=True, check=False)

    def test_report_contains_matched_fake_jobs_and_request_metrics(self):
        p = self.run_cli("--dry-run")
        self.assertEqual(p.returncode, 0, p.stderr)
        report = json.loads(p.stdout)
        self.assertEqual(report["mode"], "dry_run")
        self.assertFalse(report["live_execution_enabled"])
        self.assertEqual(report["usage_source"], "fake")
        self.assertIsNone(report["noninferiority"]["decision"])
        self.assertEqual(report["noninferiority"]["margin_percentage_points"], -3)
        self.assertEqual(report["noninferiority"]["confidence_level"], 0.95)
        self.assertEqual(report["noninferiority"]["resampling_unit"], "task")
        self.assertEqual(len(report["jobs"]), 48)
        pairs = {}
        for job in report["jobs"]:
            self.assertTrue(job["task_id"].startswith("synthetic/"))
            self.assertEqual(job["argv_preview"][:4], ["hermes", "chat", "-q", "<reviewed task prompt>"])
            self.assertIn("--oneshot", job["argv_preview"])
            self.assertEqual(job["metric_source"], "fake")
            pair = pairs.setdefault((job["task_id"], job["replicate"]), {})
            pair[job["arm"]] = job
            self.assertGreater(job["wall_ms"], 0)
            self.assertEqual(job["model_response_turns"], len(job["requests"]))
            for i, row in enumerate(job["requests"], 1):
                self.assertEqual(row["request_index"], i)
                self.assertEqual(set(row["usage"]), {
                    "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"
                })
                self.assertTrue(all(type(x) is int and x >= 0 for x in row["usage"].values()))
        for (task_id, replicate), jobs in pairs.items():
            self.assertEqual(set(jobs), set(ARMS))
            self.assertEqual({job["seed"] for job in jobs.values()}, {jobs["base"]["seed"]})
            self.assertEqual({job["execution_order"] for job in jobs.values()}, {1, 2, 3, 4})
        for task_id in {k[0] for k in pairs}:
            for arm in ARMS:
                self.assertEqual({jobs[arm]["execution_order"] for (task, _), jobs in pairs.items()
                                  if task == task_id}, {1, 2, 3, 4})
        for arm in ARMS:
            arm_jobs = [job for job in report["jobs"] if job["arm"] == arm]
            expected = sum(job["passed"] for job in arm_jobs) / len(arm_jobs)
            self.assertEqual(report["summary"][arm]["pass_rate"], expected)
        self.assertEqual(json.loads(self.run_cli("--dry-run").stdout), report)

    def test_no_live_mode_or_commands_run(self):
        for args in ((), ("--live",)):
            with self.subTest(args=args):
                p = self.run_cli(*args)
                self.assertEqual(p.returncode, 2)
                self.assertEqual(p.stdout, "")
                self.assertIn("dry-run", p.stderr)
