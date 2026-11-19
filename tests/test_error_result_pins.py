from __future__ import annotations

import unittest

from sol_pi_hermes.observation_pack import is_pure_text_tool_result


class ErrorResultPins(unittest.TestCase):
    def test_either_error_flag_excludes_result_and_clean_result_is_pure(self) -> None:
        base = {"role": "toolResult", "content": "ok"}
        self.assertTrue(is_pure_text_tool_result(base))
        self.assertFalse(is_pure_text_tool_result({**base, "isError": True}))
        self.assertFalse(is_pure_text_tool_result({**base, "is_error": True}))
        self.assertTrue(is_pure_text_tool_result({**base, "isError": False, "is_error": False}))


if __name__ == "__main__":
    unittest.main()
