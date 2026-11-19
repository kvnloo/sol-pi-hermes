from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.observation_pack import estimate_tokens, read_recall_chunk


class RecallOffsetPins(unittest.TestCase):
    def test_offset_at_size_is_empty_eof_and_past_size_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.txt"
            path.write_text("ab\n", encoding="utf-8")
            chunk = read_recall_chunk(path, 3, max_bytes=100, max_lines=10)
            self.assertEqual((chunk["text"], chunk["lines"], chunk["nextOffset"], chunk["eof"]), ("", 0, 3, True))
            with self.assertRaises(ValueError):
                read_recall_chunk(path, 4, max_bytes=100, max_lines=10)

    def test_estimate_tokens_rounds_up(self) -> None:
        self.assertEqual([estimate_tokens("x" * n) for n in (0, 1, 4, 5, 8, 9)], [0, 1, 1, 2, 2, 3])


if __name__ == "__main__":
    unittest.main()
