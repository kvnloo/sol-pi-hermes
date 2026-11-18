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


if __name__ == "__main__":
    unittest.main()
