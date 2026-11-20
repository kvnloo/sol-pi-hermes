"""Handler-level pins for plugin.register(): gates, session routing, dispatch payloads."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.observation_pack import FULL_SENDS, THRESHOLD_BYTES
from sol_pi_hermes import plugin


class _Ctx:
    def __init__(self) -> None:
        self.tools: dict = {}
        self.middleware: dict = {}
        self.dispatched: list = []

    def register_tool(self, name, toolset, schema, handler) -> None:
        self.tools[name] = handler

    def register_middleware(self, hook, fn) -> None:
        self.middleware[hook] = fn

    def dispatch_tool(self, name, payload, **context):
        self.dispatched.append((name, payload, context))
        return '{"success": true}'


def _request() -> dict:
    body = "Z\n" * (THRESHOLD_BYTES // 2 + 40)
    return {"messages": [{"role": "tool", "name": "terminal", "tool_call_id": "t", "content": body}]}


class PluginHandlerTests(unittest.TestCase):
    def _register(self, tmp: str, **cfg):
        ctx = _Ctx()
        calls: list[str] = []

        def root(sid: str) -> Path:
            calls.append(sid)
            return Path(tmp) / sid

        patcher = mock.patch.object(plugin, "runtime_root_for_session", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        plugin.register(ctx, SolPiConfig(**cfg))
        return ctx, calls

    def test_projection_is_gated_by_config_and_request_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            off, _ = self._register(tmp, observation_pack=False)
            on, _ = self._register(tmp, observation_pack=True)
            req = _request()
            outs_off = [off.middleware["llm_request"](req, session_id="s") for _ in range(FULL_SENDS + 1)]
            outs_on = [on.middleware["llm_request"](req, session_id="s") for _ in range(FULL_SENDS + 1)]
            self.assertEqual(outs_off, [None] * (FULL_SENDS + 1))
            self.assertIsNotNone(outs_on[-1])
            self.assertIsNone(on.middleware["llm_request"]("not-a-dict", session_id="s"))

    def test_obs_recall_forwards_id_and_offset_to_the_session_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, calls = self._register(tmp, observation_pack=True)
            req = _request()
            for _ in range(FULL_SENDS + 1):
                out = ctx.middleware["llm_request"](req, session_id="s")
            obs_id = out["request"]["messages"][0]["content"].split("id: ", 1)[1].split("\n", 1)[0]
            recall = ctx.tools["obs_recall"]
            first = json.loads(recall({"id": obs_id}, session_id="s"))
            later = json.loads(recall({"id": obs_id, "offset": 5}, session_id="s"))
            self.assertIn("content", first)
            self.assertEqual(later["offset"], 5)
            self.assertEqual(first["offset"], 0)
            self.assertIn("error", json.loads(recall({}, session_id="s")))
            self.assertEqual(set(calls), {"s"})
            self.assertEqual(calls.count("s"), 1)

    def test_session_id_wins_over_task_id_for_root_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, calls = self._register(tmp, observation_pack=True)
            ctx.tools["obs_recall"]({"id": "x"}, session_id="S", task_id="T")
            ctx.tools["obs_recall"]({"id": "x"}, task_id="T")
            missing = json.loads(ctx.tools["obs_recall"]({"id": "x"}))
            self.assertIn("session_id or task_id", missing["error"])
            self.assertEqual(calls, ["S", "T"])

    def test_patch_dispatch_strips_then_run_and_forwards_only_routing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx, _ = self._register(tmp, action_fusion=True)
            ctx.tools["sol_pi_patch"](
                {"path": "/a", "old_string": "x", "new_string": "y", "then_run": {"command": "true"}},
                session_id="S", task_id="T", secret="no",
            )
            name, payload, context = ctx.dispatched[0]
            self.assertEqual(name, "patch")
            self.assertEqual(payload, {"path": "/a", "old_string": "x", "new_string": "y"})
            self.assertEqual(context, {"session_id": "S", "task_id": "T"})
            ctx.dispatched.clear()
            ctx.tools["sol_pi_write"]({"path": "/a", "content": "c", "then_run": {"command": "true", "timeout": 3}})
            ctx.tools["sol_pi_write"]({"path": "/a", "content": "c", "then_run": {"command": "true"}})
            self.assertEqual(
                [(n, p) for n, p, _ in ctx.dispatched],
                [
                    ("write_file", {"path": "/a", "content": "c"}),
                    ("terminal", {"command": "true", "timeout": 3}),
                    ("write_file", {"path": "/a", "content": "c"}),
                    ("terminal", {"command": "true"}),
                ],
            )

    def test_omitted_or_null_then_run_returns_only_the_mutation_result(self) -> None:
        # Omission/null mean no follow-up. Malformed values are a separate
        # preflight-refusal contract covered by Draft #15; never pin their
        # silent removal (which would permit an unintended mutation).
        with tempfile.TemporaryDirectory() as tmp:
            ctx, _ = self._register(tmp, action_fusion=True)
            for optional in ({}, {"then_run": None}):
                with self.subTest(optional=optional):
                    ctx.dispatched.clear()
                    out_w = ctx.tools["sol_pi_write"]({"path": "/a", "content": "c", **optional})
                    out_p = ctx.tools["sol_pi_patch"]({"path": "/a", "old_string": "x", "new_string": "y", **optional})
                    self.assertEqual(out_w, '{"success": true}')
                    self.assertEqual(out_p, '{"success": true}')
                    self.assertEqual([n for n, _, _ in ctx.dispatched], ["write_file", "patch"])


if __name__ == "__main__":
    unittest.main()
