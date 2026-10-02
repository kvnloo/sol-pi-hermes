"""ActionFusion preserves the native caller and resolved mutation target."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from sol_pi_hermes.action_fusion import execute_mutation_then_run
from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.plugin import register


class ActionFusionContextTests(unittest.TestCase):
    def test_write_and_patch_forward_only_native_identity_with_legacy_compatibility(self):
        for enabled in (False, True):
            for kind in ("write", "patch"):
                for identity in ({}, {"task_id": "task-A", "session_id": "session-A"}):
                    with self.subTest(enabled=enabled, kind=kind, identity=identity):
                        with tempfile.TemporaryDirectory() as tmp:
                            target = Path(tmp) / "target.txt"
                            target.write_text("before")
                            calls = []

                            def dispatch(name, args, **kwargs):
                                calls.append((name, kwargs))
                                if name == "terminal":
                                    return json.dumps({"output": "fixture", "exit_code": 0})
                                target.write_text("after")
                                return json.dumps({"success": True, "resolved_path": str(target)})

                            ctx = MagicMock()
                            ctx.dispatch_tool.side_effect = dispatch
                            register(ctx, SolPiConfig(action_fusion=enabled))
                            handlers = {c.kwargs["name"]: c.kwargs["handler"]
                                        for c in ctx.register_tool.call_args_list}
                            args = {"path": str(target), "content": "after",
                                    "old_string": "before", "new_string": "after"}
                            for follow_up in (False, True):
                                calls.clear()
                                request = {**args, **({"then_run": {"command": "fixture"}} if follow_up else {})}
                                result = json.loads(handlers[f"sol_pi_{kind}"](
                                    request, **identity, future_context="ignored", parent_agent="not-the-owner"))
                                expected = ["write_file" if kind == "write" else "patch"]
                                if enabled and follow_up:
                                    expected.append("terminal")
                                    self.assertEqual(result["then_run"], "[then_run:succeeded]")
                                self.assertEqual([name for name, _ in calls], expected)
                                self.assertTrue(all(kwargs == identity for _, kwargs in calls))
                                self.assertEqual(target.read_text(), "after")

    def test_native_resolved_target_wins_and_legacy_receipts_keep_the_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_target, fallback = root / "native.txt", root / "fallback.txt"
            for payload in ({"resolved_path": str(native_target)}, {}, {"resolved_path": "relative.txt"},
                            {"resolved_path": None}, {"resolved_path": []}, "legacy result"):
                with self.subTest(payload=payload):
                    for path in (native_target, fallback):
                        path.unlink(missing_ok=True)
                    use_native = isinstance(payload, dict) and payload.get("resolved_path") == str(native_target)
                    expected = native_target if use_native else fallback

                    def mutate():
                        expected.write_text("after")
                        return json.dumps(payload) if isinstance(payload, dict) else payload

                    command_calls = []

                    def run(command, timeout):
                        command_calls.append((command, timeout))
                        self.assertEqual(expected.read_text(), "after")
                        return json.dumps({"output": "fixture", "exit_code": 0})

                    result = json.loads(execute_mutation_then_run(
                        mutate=mutate, absolute_path=fallback,
                        then_run={"command": "fixture"}, run_command=run))
                    self.assertEqual(result["then_run"], "[then_run:succeeded]")
                    self.assertEqual(command_calls, [("fixture", None)])


if __name__ == "__main__":
    unittest.main()
