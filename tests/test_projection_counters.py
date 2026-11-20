import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.observation_pack import FULL_SENDS, THRESHOLD_BYTES, ledger_path, project_messages

BODY = "line\n" * (THRESHOLD_BYTES // 5 + 20)


def tool():
    return {"role": "tool", "name": "terminal", "tool_call_id": "tc1", "content": BODY}


def events(root):
    return [json.loads(line) for line in ledger_path(root).read_text().splitlines()]


def is_packed(msg):
    return "[large tool result replaced" in msg["content"]


class ProjectionCounterTests(unittest.TestCase):
    def test_send_number_keeps_counting_past_the_full_sends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sent = Path(tmp), {}
            outs = [project_messages([tool()], root, sent) for _ in range(FULL_SENDS + 2)]
            self.assertEqual([is_packed(o[0]) for o in outs], [False] * FULL_SENDS + [True, True])
            self.assertEqual([e["event"] for e in events(root)], ["full"] * FULL_SENDS + ["placeholder"] * 2)
            self.assertEqual([e["sendNumber"] for e in events(root)[FULL_SENDS:]], [FULL_SENDS + 1, FULL_SENDS + 2])

    def test_prior_assistant_turns_drive_packing_and_only_assistants_count(self):
        def run(*roles):
            with tempfile.TemporaryDirectory() as tmp:
                msgs = [tool()] + [{"role": r, "content": "x"} for r in roles]
                return is_packed(project_messages(msgs, Path(tmp))[0])

        self.assertTrue(run(*["assistant"] * FULL_SENDS))
        self.assertFalse(run(*["assistant"] * (FULL_SENDS - 1)))
        self.assertFalse(run("user", "user"))
        self.assertFalse(run("assistant", "user", "user"))

    def test_ledger_request_is_trailing_assistant_count_plus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, tool(), {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "b"}]
            project_messages(msgs, root, {})
            self.assertEqual([e["request"] for e in events(root)], [3])


if __name__ == "__main__":
    unittest.main()
