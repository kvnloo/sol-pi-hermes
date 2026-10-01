from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.action_fusion import (
    THEN_RUN_FAILED,
    THEN_RUN_SKIPPED,
    THEN_RUN_SUCCEEDED,
    execute_mutation_then_run,
)


class ActionFusionStatusTests(unittest.TestCase):
    def test_returned_mutation_failure_never_runs_command_against_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.py"
            path.write_text("original contents\n")
            for failure in ({"error": "write failed"}, {"success": False, "message": "patch did not match"}):
                with self.subTest(failure=failure):
                    calls = []
                    mutation = json.dumps(failure)
                    result = json.loads(execute_mutation_then_run(
                        mutate=lambda: mutation,
                        absolute_path=path,
                        then_run={"command": "fixture-only"},
                        run_command=lambda *args: calls.append(args) or "must not run",
                    ))
                    self.assertEqual(calls, [], "failed mutation ran a chained command on old content")
                    self.assertIn(THEN_RUN_SKIPPED, result["error"])
                    self.assertEqual(result["mutation"], mutation)
                    self.assertEqual(path.read_text(), "original contents\n")
                    self.assertEqual(execute_mutation_then_run(
                        mutate=lambda: mutation, absolute_path=path, then_run=None,
                        run_command=lambda *_: self.fail("unrequested command ran"),
                    ), mutation)

    def test_command_status_reflects_returned_failure_and_keeps_output(self):
        outcomes = (
            ({"exit_code": 1, "error": None, "output": "fixture test failed"}, THEN_RUN_FAILED),
            ({"exit_code": -9, "output": "fixture command interrupted"}, THEN_RUN_FAILED),
            ({"error": "command unavailable"}, THEN_RUN_FAILED),
            ({"success": False, "message": "command failed"}, THEN_RUN_FAILED),
            ({"exit_code": 0, "error": None, "output": "the word error is only data"}, THEN_RUN_SUCCEEDED),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "changed.py"
            for payload, status in outcomes:
                with self.subTest(payload=payload):
                    calls = []
                    def mutate():
                        path.write_text("new contents\n")
                        return json.dumps({"success": True})
                    output = json.dumps(payload)
                    result = json.loads(execute_mutation_then_run(
                        mutate=mutate, absolute_path=path,
                        then_run={"command": "fixture-only", "timeout": 3},
                        run_command=lambda *args: calls.append(args) or output,
                    ))
                    self.assertEqual(calls, [("fixture-only", 3)])
                    self.assertEqual(result["then_run"], status)
                    self.assertEqual(result["output"], output)
                    if status == THEN_RUN_FAILED:
                        self.assertTrue(result["error"])
                    else:
                        self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
