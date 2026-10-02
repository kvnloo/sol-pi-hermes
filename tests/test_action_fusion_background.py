from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.action_fusion import (
    THEN_RUN_FAILED,
    THEN_RUN_PENDING,
    THEN_RUN_SUCCEEDED,
    execute_mutation_then_run,
)


class ActionFusionBackgroundTests(unittest.TestCase):
    def check_receipt(self, receipt: dict, expected: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fixture.txt"
            calls = []
            mutation = json.dumps({"success": True})
            raw_receipt = json.dumps(receipt)

            def mutate() -> str:
                target.write_text("local fixture\n")
                return mutation

            result = json.loads(execute_mutation_then_run(
                mutate=mutate,
                absolute_path=target,
                then_run={"command": "receipt-only", "timeout": 3},
                run_command=lambda *args: calls.append(args) or raw_receipt,
            ))
            self.assertEqual(calls, [("receipt-only", 3)])
            self.assertEqual(result["mutation"], mutation)
            self.assertEqual(result["output"], raw_receipt)
            self.assertEqual(target.read_text(), "local fixture\n")
            self.assertEqual(result["then_run"], expected)
            if expected == THEN_RUN_FAILED:
                self.assertTrue(result["error"])
            else:
                self.assertNotIn("error", result)

    def test_completed_foreground_success_stays_succeeded(self):
        self.check_receipt({"output": "done", "exit_code": 0, "error": None}, THEN_RUN_SUCCEEDED)

    def test_completed_foreground_failure_stays_failed(self):
        self.check_receipt({"output": "failed", "exit_code": 7, "error": None}, THEN_RUN_FAILED)

    def test_background_launch_exit_zero_is_pending(self):
        self.check_receipt({
            "output": "Background process started", "session_id": "proc_fixture",
            "exit_code": 0, "error": None,
        }, THEN_RUN_PENDING)

    def test_timeout_promotion_is_pending_regardless_of_notification_support(self):
        for notify in (True, False):
            with self.subTest(notify=notify):
                self.check_receipt({
                    "output": "Background process started", "session_id": "proc_fixture",
                    "exit_code": 0, "error": None, "notify_on_complete": notify,
                    "promoted_from_foreground": "Started in background; do not rerun.",
                }, THEN_RUN_PENDING)

    def test_user_message_yield_is_pending(self):
        self.check_receipt({
            "output": "partial", "exit_code": None, "error": None,
            "status": "yielded_to_background", "session_id": "proc_fixture",
            "notify_on_complete": True, "note": "Still running; respond to the user.",
        }, THEN_RUN_PENDING)

    def test_explicit_failure_keeps_precedence_over_background_markers(self):
        for failure in ({"error": "fixture failure"}, {"success": False, "message": "failed"}):
            with self.subTest(failure=failure):
                self.check_receipt({
                    "session_id": "proc_fixture", "status": "yielded_to_background",
                    "exit_code": 0, **failure,
                }, THEN_RUN_FAILED)

    def test_nonzero_exit_keeps_precedence_over_background_markers(self):
        for exit_code in (7, -9):
            with self.subTest(exit_code=exit_code):
                self.check_receipt({
                    "session_id": "proc_fixture", "status": "yielded_to_background",
                    "exit_code": exit_code, "error": None,
                }, THEN_RUN_FAILED)

    def test_invalid_or_empty_process_id_does_not_invent_background_receipt(self):
        for session_id in (None, "", " \t\n", 0, 1, False, True, [], ["proc_fixture"], {}, {"id": "proc_fixture"}):
            with self.subTest(session_id=session_id):
                self.check_receipt({"session_id": session_id, "exit_code": 0}, THEN_RUN_SUCCEEDED)

    def test_explicit_yield_is_pending_even_without_usable_process_id(self):
        for receipt in ({}, {"session_id": ""}, {"session_id": 123}):
            with self.subTest(receipt=receipt):
                self.check_receipt({
                    "status": "yielded_to_background", "exit_code": None, **receipt,
                }, THEN_RUN_PENDING)


if __name__ == "__main__":
    unittest.main()
