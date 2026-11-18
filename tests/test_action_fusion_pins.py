import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.action_fusion import (
    THEN_RUN_FAILED,
    THEN_RUN_SKIPPED,
    THEN_RUN_SUCCEEDED,
    THEN_RUN_UNKNOWN,
    assert_unchanged_before_command,
    file_sha256,
    execute_mutation_then_run,
)

OK = json.dumps({"success": True})


def run(mutate=lambda: OK, path=None, then_run=None, calls=None, output=json.dumps({"exit_code": 0})):
    def run_command(command, timeout):
        if calls is not None:
            calls.append((command, timeout))
        if isinstance(output, Exception):
            raise output
        return output

    return json.loads(execute_mutation_then_run(
        mutate=mutate, absolute_path=path, then_run=then_run, run_command=run_command))


class ActionFusionPins(unittest.TestCase):
    def test_command_and_timeout_are_passed_through(self):
        calls = []
        out = run(then_run={"command": "make", "timeout": 7}, calls=calls)
        self.assertEqual(calls, [("make", 7)])
        self.assertEqual(out["then_run"], THEN_RUN_SUCCEEDED)


    def test_exit_code_classification(self):
        # A negative code is still a failure.
        negative = run(then_run={"command": "x"}, output=json.dumps({"exit_code": -9}))
        self.assertEqual(negative["then_run"], THEN_RUN_FAILED)
        # JSON true is not an integer exit code: it must not be read as exit 1.
        boolean = run(then_run={"command": "x"}, output=json.dumps({"exit_code": True}))
        self.assertEqual(boolean["then_run"], THEN_RUN_UNKNOWN)

    def test_non_string_mutation_result_is_wrapped_as_json(self):
        raw = execute_mutation_then_run(
            mutate=lambda: {"a": 1}, absolute_path=None, then_run=None, run_command=lambda c, t: "")
        self.assertEqual(json.loads(raw), {"result": {"a": 1}})

    def test_mutation_exception_marks_skip_only_when_command_requested(self):
        def boom():
            raise RuntimeError("disk full")

        with_cmd = run(mutate=boom, then_run={"command": "x"})
        self.assertIn(THEN_RUN_SKIPPED, with_cmd["error"])
        self.assertIn("disk full", with_cmd["error"])
        self.assertEqual(run(mutate=boom, then_run=None), {"error": "disk full"})

    def test_unsuccessful_result_without_message_gets_default_error(self):
        out = run(mutate=lambda: json.dumps({"success": False}), then_run={"command": "x"})
        self.assertEqual(out["error"], "Tool reported an unsuccessful result")
        self.assertEqual(out["then_run"], THEN_RUN_SKIPPED)

    def test_run_command_exception_is_reported_as_failed(self):
        calls = []
        out = run(then_run={"command": "x"}, calls=calls, output=RuntimeError("no shell"))
        self.assertEqual(calls, [("x", None)])
        self.assertEqual(out["then_run"], THEN_RUN_FAILED)
        self.assertEqual(out["error"], "no shell")

    def test_host_local_path_guard_runs_command_for_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "f.txt")
            p.write_text("hi")
            calls = []
            out = run(path=p, then_run={"command": "x"}, calls=calls)
        self.assertEqual(calls, [("x", None)])
        self.assertEqual(out["then_run"], THEN_RUN_SUCCEEDED)

    def test_host_local_path_guard_blocks_command_when_target_missing(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            out = run(path=Path(d, "gone.txt"), then_run={"command": "x"}, calls=calls)
        self.assertEqual(calls, [])
        self.assertNotEqual(out["then_run"], THEN_RUN_SUCCEEDED)

    def test_assert_unchanged_before_command_direct(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "f.txt")
            with self.assertRaisesRegex(OSError, r"\[then_run:skipped\] target missing"):
                assert_unchanged_before_command(p, "0")
            p.write_text("hi")
            with self.assertRaisesRegex(OSError, r"\[then_run:skipped\] target content changed"):
                assert_unchanged_before_command(p, "0")
            self.assertIsNone(assert_unchanged_before_command(p, file_sha256(p)))


if __name__ == "__main__":
    unittest.main()
