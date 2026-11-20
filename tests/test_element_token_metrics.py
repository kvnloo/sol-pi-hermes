import json
import unittest

from sol_pi_hermes.observation_eval import evaluate_fixture


class ElementTokenMetricsTests(unittest.TestCase):
    def test_token_prefixes_do_not_count_as_visible(self):
        for short, full in (("tok_1", "tok_10"), ("s00000001:1", "s00000001:10")):
            with self.subTest(full=full):
                raw = ' { "tokens" : ' + json.dumps([full]) + ', "note": "token ' + full + '" } '
                fixture = {
                    "name": "token-boundary",
                    "tool_call_id": "offline",
                    "capture": {"elements": [
                        {"element_token": short}, {"element_token": full},
                    ]},
                    "control": {"raw_text_payload": raw},
                }
                result = evaluate_fixture(fixture)
                self.assertEqual(result["captured_element_tokens"], 2)
                self.assertEqual(result["control_element_tokens"], [full])
                for send in result["sends"]:
                    self.assertEqual(send["retained_element_tokens"], [full])
                    self.assertEqual(send["before_bytes"], len(raw.encode("utf-8")))
                    self.assertEqual(send["saved_bytes"], 0)

    def test_standalone_tokens_in_text_blocks_still_count(self):
        fixture = {
            "name": "legacy-blocks",
            "tool_call_id": "offline",
            "capture": {"elements": [{"element_token": "tok_1"}]},
            "control": {"content_blocks": [{"type": "text", "text": "token (tok_1)."}]},
        }
        result = evaluate_fixture(fixture)
        self.assertEqual(result["control_element_tokens"], ["tok_1"])
        self.assertTrue(all(s["retained_element_tokens"] == ["tok_1"] for s in result["sends"]))
