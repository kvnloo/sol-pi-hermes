"""Offline, real-host plugin discovery/request/recall contract.

Set HERMES_SOURCE to the pinned Hermes checkout. No provider, desktop, or model
is invoked. The ordinary standalone suite skips these integration tests.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from experiments.capture_result_shadow import default_fixtures

HERMES_SOURCE = os.environ.get("HERMES_SOURCE")
if HERMES_SOURCE:
    sys.path.insert(0, HERMES_SOURCE)


@unittest.skipUnless(HERMES_SOURCE, "set HERMES_SOURCE for real-host integration")
class HermesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        empty = self.base / "empty-bundled"
        empty.mkdir()
        self.enterContext(patch.dict(os.environ, {
            "HERMES_HOME": str(self.base / "unused-default"),
            "HERMES_BUNDLED_PLUGINS": str(empty),
        }))
        self.enterContext(patch.object(Path, "home", return_value=self.base))
        self.homes = [self.base / "profile-A", self.base / "profile-B"]
        for home in self.homes:
            (home / "plugins").mkdir(parents=True)
            (home / "plugins" / "sol-pi").symlink_to(
                Path(__file__).resolve().parents[1], target_is_directory=True,
            )
            (home / "config.yaml").write_text("plugins:\n  enabled: [sol-pi]\n")
            (home / "sol-pi.json").write_text('{"observationPack": true}')
        from hermes_cli.plugins import _reset_plugin_managers_for_tests
        _reset_plugin_managers_for_tests()
        self.addCleanup(_reset_plugin_managers_for_tests)

    @contextmanager
    def profile(self, home):
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            from hermes_cli.plugins import get_plugin_manager
            manager = get_plugin_manager()
            manager.discover_and_load()
            self.assertIsNone(manager._plugins["sol-pi"].error)
            yield manager
        finally:
            reset_hermes_home_override(token)

    def send(self, request, session_id):
        from hermes_cli.middleware import apply_llm_request_middleware
        return apply_llm_request_middleware(
            request, session_id=session_id, task_id="task-does-not-own-archive",
            model="offline-fixture", provider="openai", api_mode="chat_completions",
        ).payload

    def recall_all(self, obs_id, session_id):
        from tools.registry import registry
        chunks, offset = [], 0
        while True:
            row = json.loads(registry.dispatch(
                "obs_recall", {"id": obs_id, "offset": offset}, session_id=session_id,
            ))
            self.assertNotIn("error", row)
            chunks.append(row["content"].split("\n", 2)[2])
            if row["eof"]:
                return "".join(chunks)
            self.assertGreater(row["next_offset"], offset)
            offset = row["next_offset"]

    def test_discovered_plugin_projects_and_recalls_in_own_profile_and_session(self):
        """A→B→A and /new retain exact bytes with no CLI or private engine slot."""
        observations = {}
        fixtures = default_fixtures()
        for home in (*self.homes, self.homes[0]):
            with self.profile(home) as manager:
                self.assertIsNone(manager._cli_ref)
                self.assertIsNone(manager._context_engine)
                for sid in ("session-1", "session-after-new"):
                    for fixture in fixtures:
                        payload = json.loads(fixture.provider_text())
                        payload["test_provenance"] = home.name + "\n尾🙂\r\n"
                        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                        request = {
                            "model": "offline-fixture", "stream": True,
                            "extra_body": {"unchanged": [1, 2]},
                            "messages": [
                                {"role": "system", "content": [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}]},
                                {"role": "assistant", "content": None, "tool_calls": [{"id": fixture.id, "type": "function", "function": {"name": "computer_use", "arguments": "{}"}}]},
                                {"role": "tool", "name": "computer_use", "tool_call_id": fixture.id, "content": text},
                            ],
                        }
                        original = copy.deepcopy(request)
                        key = (home, sid, fixture.id)
                        if key not in observations:
                            self.assertEqual(self.send(request, sid), original)
                            self.assertEqual(self.send(request, sid), original)
                        packed = self.send(request, sid)
                        placeholder = packed["messages"][-1]["content"]
                        self.assertLess(len(placeholder.encode()), len(text.encode()))
                        obs_id = re.search(r"id: (obs_[a-f0-9]{24})", placeholder).group(1)
                        self.assertEqual(self.recall_all(obs_id, sid), text)
                        restored = copy.deepcopy(packed)
                        restored["messages"][-1]["content"] = text
                        self.assertEqual(restored, original)
                        self.assertEqual(request, original)
                        observations[key] = obs_id
                        expected = home / "sol-pi" / sid / "observation-pack" / "objects" / f"{obs_id}.txt"
                        self.assertEqual(expected.read_bytes(), text.encode())
                other = self.homes[1] if home == self.homes[0] else self.homes[0]
                if (other, "session-1", fixtures[0].id) in observations:
                    from tools.registry import registry
                    foreign = observations[(other, "session-1", fixtures[0].id)]
                    self.assertIn("error", json.loads(registry.dispatch(
                        "obs_recall", {"id": foreign}, session_id="session-1",
                    )))
        self.assertFalse((self.base / ".hermes").exists())
        self.assertFalse((self.base / "unused-default" / "sol-pi").exists())

    def test_unsupported_envelopes_and_multimodal_results_stay_unchanged(self):
        big = default_fixtures()[0].provider_text()
        cases = [
            {"input": [{"type": "function_call_output", "call_id": "x", "output": big}]},
            {"messages": [{"role": "tool", "tool_call_id": "x", "content": [{"type": "text", "text": big, "cache_control": {"type": "ephemeral"}}]}]},
            {"messages": [{"role": "tool", "tool_call_id": "x", "content": [{"type": "text", "text": big}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,c3ludGhldGlj"}}]}]},
            {"messages": [{"role": "tool", "tool_call_id": "x", "content": big, "isError": True}]},
        ]
        with self.profile(self.homes[0]) as manager:
            for request in cases:
                original = copy.deepcopy(request)
                for _ in range(4):
                    self.assertEqual(self.send(request, "session-1"), original)
                self.assertEqual(request, original)
            request = {"messages": [{"role": "tool", "content": big}]}
            from hermes_cli.middleware import apply_llm_request_middleware
            self.assertEqual(apply_llm_request_middleware(request).payload, request)
            manager.unload("sol-pi")
            self.assertEqual(self.send(request, "session-1"), request)

    def test_hermes_serialized_terminal_failure_keeps_diagnostics_inline(self):
        from tools.registry import tool_result
        from agent.tool_dispatch_helpers import make_tool_result_message
        from agent.message_metadata import without_persistence_fields

        output = "synthetic terminal diagnostic\n" * 700
        with self.profile(self.homes[0]):
            for number, (exit_code, suffix) in enumerate((
                (1, ""), (1, "\n\n[Subdirectory context discovered: synthetic test instructions]"), (0, ""),
            )):
                text = tool_result(output=output, exit_code=exit_code, error=None) + suffix
                message = make_tool_result_message("terminal", text, f"terminal-{number}")
                request = {"messages": [without_persistence_fields(message)]}
                for send in range(4):
                    projected = self.send(request, "terminal-session")
                    if exit_code or send < 2:
                        self.assertTrue(projected == request, "serialized terminal failure was packed")
                    else:
                        self.assertLess(len(projected["messages"][0]["content"]), len(text))
                self.assertEqual(request["messages"][0]["content"], text)


if __name__ == "__main__":
    unittest.main()
