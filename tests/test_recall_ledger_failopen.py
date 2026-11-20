import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.observation_pack import create_observation, ensure_stored, ledger_path, recall


class RecallLedgerFailOpenTests(unittest.TestCase):
    def test_symlinked_ledger_does_not_fail_successful_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            observation = create_observation({"role": "tool", "content": "payload\n" * 3000}, root)
            ensure_stored(observation)
            target = Path(tmp) / "elsewhere"
            ledger_path(root).symlink_to(target)
            result = json.loads(recall(root, observation.id))
            self.assertNotIn("error", result)
            self.assertIn("payload", result["content"])
            self.assertFalse(target.exists())

    def test_healthy_recall_still_writes_one_ledger_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            observation = create_observation({"role": "tool", "content": "payload\n" * 3000}, root)
            ensure_stored(observation)
            result = json.loads(recall(root, observation.id))
            lines = [json.loads(l) for l in ledger_path(root).read_text().splitlines()]
            records = [r for r in lines if r["event"] == "recall"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["id"], observation.id)
            self.assertEqual(records[0]["nextOffset"], result["next_offset"])

    def test_unwritable_ledger_does_not_fail_successful_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            observation = create_observation({"role": "tool", "content": "payload\n" * 3000}, root)
            ensure_stored(observation)
            ledger_path(root).mkdir()  # open(2) on a directory raises IsADirectoryError
            result = json.loads(recall(root, observation.id))
            self.assertNotIn("error", result)
            self.assertIn("payload", result["content"])


if __name__ == "__main__":
    unittest.main()
