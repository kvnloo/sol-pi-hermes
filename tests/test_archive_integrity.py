"""An observation may be packed only after its complete archive is written."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.observation_pack import (
    create_observation,
    project_messages,
    recall,
)


def old_history(body: str) -> list[dict]:
    # A restarted plugin can first encounter an observation after two replies.
    return [
        {"role": "user", "content": "Inspect the output"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "archive-test", "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }]},
        {"role": "tool", "name": "terminal", "tool_call_id": "archive-test", "content": body},
        {"role": "assistant", "content": "I have the output"},
        {"role": "user", "content": "Keep checking"},
        {"role": "assistant", "content": "I am checking it"},
        {"role": "user", "content": "Continue"},
    ]


class ArchiveIntegrityTests(unittest.TestCase):
    def test_short_writes_preserve_complete_archive_and_paged_recall(self):
        body = "archive evidence: 尾🙂\n" * 900
        real_write = os.write
        with tempfile.TemporaryDirectory() as tmp:
            for write_limit in (len(body.encode("utf-8")), 127):
                with self.subTest(write_limit=write_limit):
                    root = Path(tmp) / str(write_limit)
                    history = old_history(body)
                    with patch("sol_pi_hermes.observation_pack.os.write", side_effect=(
                        lambda fd, data: real_write(fd, data[:write_limit])
                    )):
                        projected = project_messages(history, root, {})
                    self.assertNotEqual(projected[2]["content"], body)
                    self.assertEqual(history[2]["content"], body)
                    observation = create_observation(history[2], root)
                    archived = observation.file_path.read_bytes()
                    self.assertEqual(len(archived), len(body.encode("utf-8")))
                    self.assertEqual(archived, body.encode("utf-8"))
                    pieces, offset = [], 0
                    while True:
                        result = json.loads(recall(root, observation.id, offset))
                        self.assertNotIn("error", result)
                        pieces.append(result["content"].split("\n", 2)[2])
                        if result["eof"]:
                            break
                        self.assertGreater(result["next_offset"], offset)
                        offset = result["next_offset"]
                    self.assertEqual("".join(pieces), body)

    def test_write_without_progress_keeps_observation_inline_and_uncounted(self):
        body = "archive evidence: 尾🙂\n" * 900
        history = old_history(body)
        counts = {}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sol_pi_hermes.observation_pack.os.write", side_effect=(
                0, AssertionError("A zero-progress write must not be retried"),
            )) as write:
                projected = project_messages(history, Path(tmp), counts)
            write.assert_called_once()
            self.assertEqual(projected, history)
            self.assertEqual(counts, {})


if __name__ == "__main__":
    unittest.main()
