from __future__ import annotations

import unittest

from sol_pi_hermes.observation_pack import count_lines


class CountLinesTests(unittest.TestCase):
    def test_counts_terminated_and_unterminated_lines(self) -> None:
        self.assertEqual(count_lines(""), 0)
        self.assertEqual(count_lines("\n"), 1)
        self.assertEqual(count_lines("a"), 1)
        self.assertEqual(count_lines("a\n"), 1)
        self.assertEqual(count_lines("a\nb"), 2)
        self.assertEqual(count_lines("a\nb\n"), 2)
        self.assertEqual(count_lines("a\n\nb\n"), 3)


if __name__ == "__main__":
    unittest.main()
