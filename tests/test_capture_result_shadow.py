from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.capture_result_shadow import default_fixtures, evaluate_fixture, run


class CaptureResultShadowTests(unittest.TestCase):
    def test_default_matrix_covers_requested_failure_shapes(self):
        ids = {f.id for f in default_fixtures()}
        self.assertEqual(
            ids,
            {"large_tree", "modal", "ambiguous", "stale_tokens", "repeated"},
        )

    def test_first_two_sends_are_exact_and_recall_recovers_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_fixture(default_fixtures()[0], Path(tmp))
        self.assertTrue(result["full_send_1_unchanged"])
        self.assertTrue(result["full_send_2_unchanged"])
        self.assertLess(result["placeholder_bytes"], result["original_bytes"])
        self.assertEqual(result["recall_recovery_rate"], 1.0)

    def test_full_shadow_matrix_is_recoverable_without_claiming_placeholder_equivalence(self):
        report = run()
        self.assertTrue(report["all_first_two_unchanged"])
        self.assertTrue(report["all_markers_recoverable"])
        self.assertTrue(report["placeholder_retention_is_not_equivalence"])
        self.assertTrue(
            any(row["placeholder_retention_rate"] < 1.0 for row in report["rows"]),
            "fixture must expose that head/tail placeholder can drop middle evidence",
        )


if __name__ == "__main__":
    unittest.main()
