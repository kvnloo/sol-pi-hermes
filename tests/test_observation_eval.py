"""Contract tests for the offline Hermes CaptureResult / ObservationPack shadow."""

from __future__ import annotations

import copy
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from sol_pi_hermes.observation_eval import (
    DEFAULT_FIXTURES,
    evaluate_fixture,
    evaluate_recordings,
    main,
)


def _fixture_set() -> dict:
    return json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8"))


def _deterministic_clock():
    return itertools.count(0, 100).__next__


class ObservationShadowTests(unittest.TestCase):
    def test_recorded_control_survives_replay_and_full_recall(self):
        frozen = _fixture_set()
        original = copy.deepcopy(frozen)
        report = evaluate_recordings(frozen, clock=_deterministic_clock())

        self.assertEqual(frozen, original)
        self.assertEqual(
            {case["name"] for case in report["cases"]},
            {"large_tree", "modal", "ambiguous", "stale", "repeated", "small", "multimodal"},
        )
        self.assertEqual(report["host_main_sha"], frozen["host_main_sha"])
        self.assertGreater(report["totals"]["saved_bytes"], 0)
        self.assertGreater(report["totals"]["saved_estimated_text_tokens"], 0)
        self.assertEqual(report["totals"]["lost_fields_after_recall"], [])
        self.assertTrue(all(case["control_unchanged"] for case in report["cases"]))
        self.assertTrue(all(send["latency_ns"] == 100 for case in report["cases"] for send in case["sends"]))

    def test_large_tree_modal_and_ambiguous_hide_structured_fields_but_recall_exactly(self):
        fixtures = {f["name"]: f for f in _fixture_set()["cases"]}
        for name in ("large_tree", "modal", "ambiguous", "stale", "repeated"):
            with self.subTest(name=name):
                case = evaluate_fixture(fixtures[name], clock=_deterministic_clock())
                first, second, third = case["sends"]
                self.assertFalse(first["packed"])
                self.assertFalse(second["packed"])
                self.assertEqual(first["saved_bytes"], 0)
                self.assertEqual(second["saved_bytes"], 0)
                self.assertTrue(third["packed"])
                self.assertGreater(third["saved_bytes"], 0)
                self.assertGreater(third["saved_estimated_text_tokens"], 0)
                self.assertEqual(third["byte_exact_recall"], True)
                self.assertEqual(third["lost_fields_after_recall"], [])
                self.assertIn("elements", third["visible_missing_fields"])
                self.assertIn("summary", third["visible_missing_fields"])
                self.assertEqual(third["retained_indices"], [])
                self.assertEqual(len(first["retained_indices"]), case["captured_element_count"])
                self.assertEqual(first["retained_indices"], second["retained_indices"])

        self.assertEqual(fixtures["modal"]["capture"]["elements"][0]["role"], "AXDialog")
        self.assertEqual(fixtures["ambiguous"]["capture"]["elements"][0]["label"], "Open")
        self.assertEqual(fixtures["ambiguous"]["capture"]["elements"][1]["label"], "Open")

    def test_capture_only_stale_tokens_are_not_misreported_as_pack_loss(self):
        fixture = next(x for x in _fixture_set()["cases"] if x["name"] == "stale")
        case = evaluate_fixture(fixture, clock=_deterministic_clock())
        self.assertGreater(case["captured_element_tokens"], 0)
        # Hermes _element_to_dict has no element_token field, before packing.
        self.assertEqual(case["control_element_tokens"], [])
        for send in case["sends"]:
            self.assertEqual(send["retained_element_tokens"], [])
            self.assertEqual(send["lost_fields_after_recall"], [])

    def test_small_and_image_bearing_captures_are_not_projected(self):
        fixtures = {x["name"]: x for x in _fixture_set()["cases"]}
        for name in ("small", "multimodal"):
            with self.subTest(name=name):
                case = evaluate_fixture(fixtures[name], clock=_deterministic_clock())
                self.assertEqual(case["saved_bytes"], 0)
                self.assertEqual(case["saved_estimated_text_tokens"], 0)
                self.assertEqual(case["sends"][0]["packed"], False)
                self.assertTrue(all(not x["packed"] for x in case["sends"]))
                self.assertTrue(all(x["visible_missing_fields"] == [] for x in case["sends"]))
                self.assertTrue(case["control_unchanged"])

    def test_invalid_fixtures_fail_instead_of_reporting_false_savings(self):
        fixtures = _fixture_set()
        bad = copy.deepcopy(fixtures)
        bad["cases"][0]["capture"]["elements"] = None
        with self.assertRaises(ValueError):
            evaluate_recordings(bad, clock=_deterministic_clock())

        duplicate = copy.deepcopy(fixtures)
        duplicate["cases"][1]["name"] = duplicate["cases"][0]["name"]
        with self.assertRaises(ValueError):
            evaluate_recordings(duplicate, clock=_deterministic_clock())

        malformed = copy.deepcopy(fixtures["cases"][0])
        malformed["control"] = {"text_payload": {}, "content_blocks": []}
        with self.assertRaises(ValueError):
            evaluate_fixture(malformed, clock=_deterministic_clock())

    def test_cli_emits_machine_readable_report_without_an_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            out = StringIO()
            with redirect_stdout(out):
                exit_code = main(["--fixtures", str(DEFAULT_FIXTURES), "--output", str(report_path)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(out.getvalue()), json.loads(report_path.read_text()))
            self.assertEqual(json.loads(out.getvalue())["totals"]["lost_fields_after_recall"], [])


if __name__ == "__main__":
    unittest.main()
