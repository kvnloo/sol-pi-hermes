"""Regression coverage for SoL-Pi's session-scoped ObservationPack storage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.observation_pack import (
    FULL_SENDS,
    THRESHOLD_BYTES,
    create_observation,
    ensure_stored,
    project_request,
    recall,
    runtime_root_for_session,
)
from sol_pi_hermes.plugin import register


class _Context:
    def __init__(self):
        self.tools = {}

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]

    def register_middleware(self, *_args):
        pass


class SessionIsolationTests(unittest.TestCase):
    def test_colliding_session_slugs_cannot_recall_each_others_data(self):
        # Previously both raw IDs resolved to the same "tenant_a" directory.
        owner_id, other_id = "tenant/a", "tenant:a"
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            observation = create_observation(
                {
                    "role": "tool",
                    "name": "terminal",
                    "tool_call_id": "private-result",
                    "content": "private session output\n" + "Z" * (THRESHOLD_BYTES + 1),
                },
                runtime_root_for_session(owner_id, home),
            )
            self.assertIsNotNone(observation)
            ensure_stored(observation)

            ctx = _Context()
            with patch(
                "sol_pi_hermes.plugin.runtime_root_for_session",
                side_effect=lambda sid: runtime_root_for_session(sid, home),
            ):
                register(ctx, SolPiConfig(observation_pack=True))
                owner_result = json.loads(
                    ctx.tools["obs_recall"]({"id": observation.id}, session_id=owner_id)
                )
                other_result = json.loads(
                    ctx.tools["obs_recall"]({"id": observation.id}, session_id=other_id)
                )

            self.assertIn("private session output", owner_result["content"])
            self.assertEqual(
                other_result, {"error": f"Unknown observation id: {observation.id}"}
            )
            self.assertNotEqual(
                runtime_root_for_session(owner_id, home),
                runtime_root_for_session(other_id, home),
            )

    def test_untrusted_session_ids_remain_in_one_bounded_path_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            ids = [
                "../../../../tmp", "tenant_a", "tenant/a", "tenant:a",
                "tenant\\a", "☃" * 500, "x" * 1000, "", "a\ud800b", "a\udc00b",
            ]
            paths = [runtime_root_for_session(value, home) for value in ids]
            self.assertEqual(len(paths), len(set(paths)))
            for path in paths:
                self.assertEqual(path.parent, home / "sol-pi")
                self.assertLessEqual(len(path.name), 105)
                self.assertRegex(path.name, r"^[A-Za-z0-9_-]+-[0-9a-f]{64}$")

    def test_retained_transcript_rearchives_under_new_root(self):
        # README upgrade claim: old slug-only archives are ignored, and the
        # transcript's full observation is archived again under the new root.
        body = "retained output\n" * THRESHOLD_BYTES
        messages = [{"role": "tool", "name": "terminal", "tool_call_id": "t", "content": body}]
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            legacy = home / "sol-pi" / "tenant_a"
            root = runtime_root_for_session("tenant/a", home)
            old = create_observation(messages[0], legacy)
            ensure_stored(old)
            self.assertIn("error", json.loads(recall(root, old.id, 0)))
            sent = {}
            for _ in range(FULL_SENDS + 1):
                out = project_request({"messages": messages}, root, sent)
            packed = out["request"]["messages"][0]["content"]
            obs_id = packed.split("id: ", 1)[1].split("\n", 1)[0]
            self.assertEqual(obs_id, old.id)
            self.assertIn("retained output", json.loads(recall(root, obs_id, 0))["content"])
            self.assertEqual(messages[0]["content"], body)


if __name__ == "__main__":
    unittest.main()
