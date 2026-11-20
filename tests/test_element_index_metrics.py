import json
import unittest

from sol_pi_hermes.observation_eval import _indices, evaluate_fixture


class ElementIndexMetricsTests(unittest.TestCase):
    def test_formatted_raw_control_counts_only_actual_element_indices(self):
        raw = r' { "elements": [{"\u0069ndex":7},{"index":3},{"index":3}],"metadata":{"index":99}} '
        fixture = {
            "name": "raw-indices",
            "tool_call_id": "offline",
            "capture": {"elements": [{"index": 3}, {"index": 7}]},
            "control": {"raw_text_payload": raw},
        }
        result = evaluate_fixture(fixture)
        self.assertEqual(result["control_indices"], [3, 7])
        for send in result["sends"]:
            self.assertEqual(send["retained_indices"], [3, 7])
            self.assertEqual(send["before_bytes"], len(raw.encode("utf-8")))
            self.assertFalse(send["packed"])

    def test_structured_indices_ignore_noninteger_entries(self):
        text = json.dumps({"elements": [
            None, {"index": True}, {"index": "4"}, {"index": 8.5},
            {"index": -4}, {"index": 4}, {"index": 0},
        ]})
        self.assertEqual(_indices(text), [0, 4])

    def test_non_structured_text_keeps_legacy_excerpt_heuristic(self):
        self.assertEqual(_indices('packed excerpt\n  {"index": 7,\n'), [7])
        self.assertEqual(_indices('{"legacy":{"index":9}}'), [9])
