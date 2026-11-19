"""Contract tests for the offline Hermes CaptureResult / ObservationPack shadow."""

from __future__ import annotations

import copy
import itertools
import json
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from sol_pi_hermes import observation_eval
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

    def test_text_block_control_reports_hidden_fields_like_text_payload(self):
        large = {f["name"]: f for f in _fixture_set()["cases"]}["large_tree"]
        blocks = copy.deepcopy(large)
        blocks["control"] = {"content_blocks": [
            {"type": "text", "text": json.dumps(large["control"]["text_payload"])}]}
        got = evaluate_fixture(blocks, clock=_deterministic_clock())["sends"][2]
        want = evaluate_fixture(large, clock=_deterministic_clock())["sends"][2]
        self.assertTrue(got["packed"])
        self.assertIn("elements", got["visible_missing_fields"])
        self.assertEqual(got["visible_missing_fields"], want["visible_missing_fields"])

    def test_shadow_eval_never_touches_the_hermes_home(self):
        with tempfile.TemporaryDirectory() as home:
            hostile = Path(home) / ".hermes"
            hostile.mkdir()
            before = sorted(p.name for p in hostile.rglob("*"))
            with mock.patch.object(Path, "home", return_value=Path(home)), mock.patch.dict(
                "os.environ", {"HOME": home, "HERMES_HOME": str(hostile)}
            ):
                report = evaluate_recordings(_fixture_set(), clock=_deterministic_clock())
            self.assertEqual(sorted(p.name for p in hostile.rglob("*")), before)
        packed = {c["name"]: [s["packed"] for s in c["sends"]] for c in report["cases"]}
        self.assertEqual(packed["large_tree"], [False, False, True])
        self.assertEqual(packed["repeated"], [False, False, True])

    def test_content_blocks_control_is_copied_not_aliased_from_fixture(self):
        multimodal = {f["name"]: f for f in _fixture_set()["cases"]}["multimodal"]
        before = copy.deepcopy(multimodal)

        def scribble(request, root, sent_counts):
            request["messages"][0]["content"][0]["scribble"] = True
            return None

        with mock.patch.object(observation_eval, "project_request", scribble):
            evaluate_fixture(multimodal, clock=_deterministic_clock())
        self.assertEqual(multimodal, before)

    def test_large_tree_and_repeated_hide_structured_fields_but_recall_exactly(self):
        fixtures = {f["name"]: f for f in _fixture_set()["cases"]}
        for name in ("large_tree", "repeated"):
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

    def test_small_modal_ambiguous_stale_and_image_captures_pass_through(self):
        fixtures = {x["name"]: x for x in _fixture_set()["cases"]}
        for name in ("small", "modal", "ambiguous", "stale", "multimodal"):
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

    def test_report_totals_and_provenance_match_per_send_rows(self):
        frozen = _fixture_set()
        report = evaluate_recordings(frozen, clock=_deterministic_clock())
        rows = [send for case in report["cases"] for send in case["sends"]]
        totals = report["totals"]

        self.assertEqual(report["fixture_provenance"], frozen["provenance"])
        self.assertEqual(totals["before_bytes"], sum(r["before_bytes"] for r in rows))
        self.assertEqual(totals["after_bytes"], sum(r["after_bytes"] for r in rows))
        self.assertLess(totals["after_bytes"], totals["before_bytes"])
        self.assertEqual(totals["projection_latency_ns"], sum(r["latency_ns"] for r in rows))
        self.assertEqual(totals["saved_estimated_text_tokens"],
                         sum(r["saved_estimated_text_tokens"] for r in rows))
        self.assertGreater(totals["projection_latency_ns"], 0)

    def test_nonempty_non_list_cases_rejected_as_malformed_schema(self):
        fixtures = _fixture_set()
        fixtures["cases"] = {"name": "not-a-list"}
        with self.assertRaisesRegex(ValueError, "cases array"):
            evaluate_recordings(fixtures, clock=_deterministic_clock())

    def test_cli_rejects_empty_replay_without_writing_success_report(self):
        import subprocess
        import sys

        fixtures = _fixture_set()
        fixtures["cases"] = []
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "empty.json"
            report_path = Path(tmp) / "report.json"
            fixture_path.write_text(json.dumps(fixtures), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "sol_pi_hermes.observation_eval",
                 "--fixtures", str(fixture_path), "--output", str(report_path)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("cases must contain at least one fixture", result.stderr)
            self.assertFalse(report_path.exists())

    def test_cli_emits_machine_readable_report_without_an_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            out = StringIO()
            with redirect_stdout(out):
                exit_code = main(["--fixtures", str(DEFAULT_FIXTURES), "--output", str(report_path)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(out.getvalue()), json.loads(report_path.read_text()))
            self.assertEqual(json.loads(out.getvalue())["totals"]["lost_fields_after_recall"], [])

    def test_cli_report_is_sorted_key_indented_and_file_matches_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            out = StringIO()
            with redirect_stdout(out):
                main(["--fixtures", str(DEFAULT_FIXTURES), "--output", str(report_path)])
            text = out.getvalue()
            canonical = json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
            self.assertEqual(text, canonical)
            self.assertEqual(report_path.read_text(), canonical)

    def test_rejects_wrong_schema_version_or_non_list_cases(self):
        for bad in ({"schema_version": 2, "cases": []}, {"schema_version": 1, "cases": {}}):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                ValueError, "expected schema_version=1 and cases array"
            ):
                evaluate_recordings(bad)

    def test_cli_fails_on_inexact_recall_even_when_no_field_is_lost(self):
        from unittest import mock
        from sol_pi_hermes import observation_eval

        real = observation_eval.recall

        def padded(*args):
            page = json.loads(real(*args))
            if page["eof"]:
                page["content"] += " "  # JSON-equivalent but not byte exact
            return json.dumps(page)

        with mock.patch.object(observation_eval, "recall", padded), redirect_stdout(StringIO()) as out:
            exit_code = main(["--fixtures", str(DEFAULT_FIXTURES)])
        report = json.loads(out.getvalue())
        self.assertEqual(report["totals"]["lost_fields_after_recall"], [])
        self.assertTrue(any(s["byte_exact_recall"] is False for c in report["cases"] for s in c["sends"]))
        self.assertEqual(exit_code, 1)

    def test_cli_fails_on_lost_field_even_when_recall_is_byte_exact(self):
        with mock.patch.object(observation_eval, "_missing_fields", return_value=["app"]), redirect_stdout(
            StringIO()
        ) as out:
            exit_code = main(["--fixtures", str(DEFAULT_FIXTURES)])
        report = json.loads(out.getvalue())
        self.assertIn("app", report["totals"]["lost_fields_after_recall"])
        self.assertTrue(
            all(s["byte_exact_recall"] is not False for c in report["cases"] for s in c["sends"])
        )
        self.assertEqual(exit_code, 1)

    def test_cli_fails_when_recall_loses_a_field(self):
        from unittest import mock
        from sol_pi_hermes import observation_eval

        real = observation_eval.recall

        def renamed(*args):
            page = json.loads(real(*args))
            page["content"] = page["content"].replace('"app"', '"zpp"', 1)
            return json.dumps(page)

        with mock.patch.object(observation_eval, "recall", renamed), redirect_stdout(StringIO()) as out:
            exit_code = main(["--fixtures", str(DEFAULT_FIXTURES)])
        self.assertIn("app", json.loads(out.getvalue())["totals"]["lost_fields_after_recall"])
        self.assertEqual(exit_code, 1)

    def test_non_object_json_control_reports_content_lost_after_recall(self):
        case = copy.deepcopy(_fixture_set()["cases"][0])
        text = json.dumps([case["control"]["text_payload"]])
        case["control"] = {"content_blocks": [{"type": "text", "text": text}]}
        for label, recalled in (("not json", "}{"), ("other json", "{}"), ("other array", "[1]")):
            with self.subTest(label), mock.patch.object(
                observation_eval, "_recall_exact_text", return_value=recalled
            ):
                third = evaluate_fixture(case, clock=_deterministic_clock())["sends"][2]
                self.assertTrue(third["packed"])
                self.assertIs(third["byte_exact_recall"], False)
                self.assertEqual(third["lost_fields_after_recall"], ["<content>"])

    def test_object_control_recalled_as_non_object_reports_content_lost(self):
        case = _fixture_set()["cases"][0]
        with mock.patch.object(observation_eval, "_recall_exact_text", return_value="[1]"):
            third = evaluate_fixture(case, clock=_deterministic_clock())["sends"][2]
        self.assertEqual(third["lost_fields_after_recall"], ["<content>"])

    def test_control_with_neither_payload_form_is_a_named_error(self):
        # Distinct from the top-level required keys: control present but empty.
        fixture = copy.deepcopy(_fixture_set()["cases"][0])
        fixture["control"] = {}
        with self.assertRaisesRegex(ValueError, "exactly one of text_payload, raw_text_payload or content_blocks"):
            evaluate_fixture(fixture, clock=_deterministic_clock())

    def test_report_states_shadow_method_and_token_estimator(self):
        report = evaluate_recordings(_fixture_set(), clock=_deterministic_clock())
        self.assertEqual(
            report["method"],
            "request-time shadow only; original provider-visible control retained",
        )
        self.assertTrue(report["token_estimator"].startswith("ceil(text characters / 4)"))

    def test_report_names_the_observation_id_only_for_packed_sends(self):
        fixture = next(f for f in _fixture_set()["cases"] if f["name"] == "large_tree")
        sends = evaluate_fixture(fixture, clock=_deterministic_clock())["sends"]
        for send in sends:
            if send["packed"]:
                self.assertRegex(send["observation_id"], r"^obs_[a-f0-9]{24}$")
            else:
                self.assertIsNone(send["observation_id"])
        self.assertTrue(any(send["packed"] for send in sends))

    def test_changed_capture_without_observation_id_is_a_named_error(self):
        fixture = next(f for f in _fixture_set()["cases"] if f["name"] == "large_tree")

        def stripped(request, root, counts):
            changed = copy.deepcopy(request)
            changed["messages"][0]["content"] = "no archive header"
            return {"request": changed}

        with mock.patch.object(observation_eval, "project_request", stripped):
            with self.assertRaisesRegex(ValueError, "large_tree: changed capture lacks an observation id"):
                evaluate_fixture(fixture, clock=_deterministic_clock())

    def test_cli_fails_when_projection_packs_before_full_sends(self):
        from unittest import mock
        from sol_pi_hermes import observation_eval

        real = observation_eval.project_request

        def premature(request, root, counts):
            # Simulate premature count advancement without damaging the archive
            # or mutating the original request.
            for _ in range(observation_eval.FULL_SENDS):
                real(request, root, counts)
            return real(request, root, counts)

        with mock.patch.object(observation_eval, "project_request", premature), redirect_stdout(StringIO()) as out:
            exit_code = main(["--fixtures", str(DEFAULT_FIXTURES)])
        cases = json.loads(out.getvalue())["cases"]
        self.assertTrue(any(c["sends"][0]["packed"] for c in cases))
        self.assertTrue(all(c["control_unchanged"] for c in cases))
        self.assertTrue(all(s["byte_exact_recall"] is not False for c in cases for s in c["sends"]))
        self.assertEqual(exit_code, 1)

    def test_cli_fails_when_projection_packs_on_the_last_full_send(self):
        from unittest import mock
        from sol_pi_hermes import observation_eval

        real = observation_eval.project_request
        calls = {"n": 0}

        def premature_on_second_send(request, root, counts):
            calls["n"] += 1
            if calls["n"] % (observation_eval.FULL_SENDS + 1) == observation_eval.FULL_SENDS:
                real(request, root, counts)  # advance the count early on the final full send only
            return real(request, root, counts)

        with mock.patch.object(observation_eval, "project_request", premature_on_second_send), redirect_stdout(StringIO()) as out:
            exit_code = main(["--fixtures", str(DEFAULT_FIXTURES)])
        sends = json.loads(out.getvalue())["cases"][0]["sends"]
        self.assertFalse(sends[0]["packed"])
        self.assertTrue(sends[observation_eval.FULL_SENDS - 1]["packed"])
        self.assertEqual(exit_code, 1)

    def test_cli_rejects_image_loss_even_when_text_recall_is_exact(self):
        fixtures = _fixture_set()
        image_case = next(c for c in fixtures["cases"] if c["name"] == "multimodal")
        image_case["control"]["content_blocks"][0]["text"] += "x" * 20000
        fixtures["cases"] = [image_case]
        real = observation_eval.project_request

        def strip_image(request, root, counts):
            text_only = copy.deepcopy(request)
            text_only["messages"][0]["content"] = request["messages"][0]["content"][0]["text"]
            return real(text_only, root, counts)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            path.write_text(json.dumps(fixtures))
            with mock.patch.object(observation_eval, "project_request", strip_image), redirect_stdout(StringIO()) as out:
                exit_code = main(["--fixtures", str(path)])
        case = json.loads(out.getvalue())["cases"][0]
        self.assertTrue(case["control_unchanged"])
        self.assertFalse(case["sends"][0]["packed"])
        self.assertFalse(case["sends"][1]["packed"])
        self.assertTrue(case["sends"][2]["byte_exact_recall"])
        self.assertFalse(case["non_text_content_unchanged"])
        self.assertEqual(exit_code, 1)

    def test_cli_fails_when_projection_mutates_the_control(self):
        from unittest import mock
        from sol_pi_hermes import observation_eval

        real = observation_eval.project_request

        def mutate(request, *args):
            result = real(request, *args)
            request["messages"][0]["tampered"] = True
            return result

        with mock.patch.object(observation_eval, "project_request", mutate), redirect_stdout(StringIO()) as out:
            exit_code = main(["--fixtures", str(DEFAULT_FIXTURES)])
        self.assertFalse(any(c["control_unchanged"] for c in json.loads(out.getvalue())["cases"]))
        self.assertEqual(exit_code, 1)

    def test_missing_required_fields_raise_named_value_errors(self):
        for key in ("host_main_sha", "provenance"):
            bad = _fixture_set()
            del bad[key]
            with self.assertRaisesRegex(ValueError, key):
                evaluate_recordings(bad)
        for key in ("name", "capture", "control", "tool_call_id"):
            case = copy.deepcopy(_fixture_set()["cases"][0])
            del case[key]
            with self.assertRaisesRegex(ValueError, key):
                evaluate_fixture(case)

    def test_invalid_recordings_are_rejected(self):
        bad_schema = {**_fixture_set(), "schema_version": 2}
        with self.assertRaises(ValueError):
            evaluate_recordings(bad_schema)
        case = copy.deepcopy(_fixture_set()["cases"][0])
        case["name"] = ""
        with self.assertRaises(ValueError):
            evaluate_fixture(case)

    def test_totals_percent_and_multimodal_bytes_count_the_image(self):
        report = evaluate_recordings(_fixture_set(), clock=_deterministic_clock())
        totals = report["totals"]
        self.assertEqual(totals["saved_percent"], round(100 * totals["saved_bytes"] / totals["before_bytes"], 2))
        self.assertGreater(totals["saved_percent"], 0)
        image = next(c for c in report["cases"] if c["name"] == "multimodal")
        blocks = next(c for c in _fixture_set()["cases"] if c["name"] == "multimodal")["control"]["content_blocks"]
        self.assertEqual(
            image["sends"][0]["before_bytes"],
            len(json.dumps(blocks, ensure_ascii=False, sort_keys=True).encode("utf-8")),
        )
        self.assertEqual(image["sends"][0]["before_bytes"], image["sends"][0]["after_bytes"])

    def test_saved_percent_uses_exact_before_bytes_denominator(self):
        real = observation_eval.project_request

        def inflate(request, root, counts):
            out = real(request, root, counts)
            if out is not None:
                out["request"]["messages"][0]["content"] += "\n" + "x" * 1_000_000
            return out

        frozen = _fixture_set()
        case = next(x for x in frozen["cases"] if x["name"] == "large_tree")
        with mock.patch.object(observation_eval, "project_request", inflate):
            totals = evaluate_recordings({**frozen, "cases": [case]})["totals"]
        # 66453 control bytes; one packed send saves 21728 and pays a 1_000_001-byte pad.
        self.assertEqual((totals["before_bytes"], totals["saved_bytes"]), (66453, -978273))
        self.assertEqual(totals["saved_percent"], -1472.13)

    def test_multimodal_wire_bytes_count_non_ascii_text_unescaped(self):
        fixture = copy.deepcopy(next(c for c in _fixture_set()["cases"] if c["name"] == "multimodal"))
        fixture["control"]["content_blocks"] = [{"type": "text", "text": "caf\u00e9"}]
        send = evaluate_fixture(fixture, clock=_deterministic_clock())["sends"][0]
        # Hand-computed UTF-8 length of [{"text": "café", "type": "text"}] (no \u escapes).
        self.assertEqual(send["before_bytes"], 35)

    def test_control_text_joins_blocks_with_newline_and_treats_missing_text_as_empty(self):
        fixture = copy.deepcopy(next(c for c in _fixture_set()["cases"] if c["name"] == "multimodal"))
        for blocks, expected in (
            (["ab", {"type": "text", "text": "cd"}], 2),  # "ab\ncd" is 5 chars; "abcd" would be 1
            ([{"type": "text"}], 0),  # absent text contributes "", not a placeholder
        ):
            with self.subTest(blocks=blocks):
                fixture["control"]["content_blocks"] = blocks
                send = evaluate_fixture(fixture, clock=_deterministic_clock())["sends"][0]
                self.assertEqual(send["before_estimated_text_tokens"], expected)

    def test_control_indices_and_visible_element_tokens_are_reported(self):
        fixture = copy.deepcopy(next(x for x in _fixture_set()["cases"] if x["name"] == "stale"))
        token = fixture["capture"]["elements"][0]["element_token"]
        fixture["control"]["text_payload"]["note"] = f"token {token}"
        case = evaluate_fixture(fixture, clock=_deterministic_clock())
        self.assertEqual(case["control_element_tokens"], [token])
        self.assertEqual(case["control_indices"], list(range(1, case["captured_element_count"] + 1)))
        for send in case["sends"]:
            self.assertEqual(send["retained_element_tokens"], [token])

    def test_empty_element_token_is_not_reported_as_a_retained_token(self):
        fixture = copy.deepcopy(next(x for x in _fixture_set()["cases"] if x["name"] == "stale"))
        fixture["capture"]["elements"][0]["element_token"] = ""
        case = evaluate_fixture(fixture, clock=_deterministic_clock())
        self.assertNotIn("", case["control_element_tokens"])
        for send in case["sends"]:
            self.assertNotIn("", send["retained_element_tokens"])

    def test_changed_field_value_counts_as_lost_not_only_removed_keys(self):
        lost = observation_eval._missing_fields(
            '{"a": 1, "b": 2, "c": 3}', '{"a": 1, "b": 99}')
        self.assertEqual(lost, ["b", "c"])

    def test_inflating_projection_reports_negative_savings(self):
        real = observation_eval.project_request

        def inflate(request, root, counts):
            out = real(request, root, counts)
            if out is not None:
                message = out["request"]["messages"][0]
                message["content"] += "\n" + "x" * 50_000
            return out

        fixture = next(x for x in _fixture_set()["cases"] if x["name"] == "large_tree")
        with mock.patch.object(observation_eval, "project_request", inflate):
            case = evaluate_fixture(fixture, clock=_deterministic_clock())
        packed = [s for s in case["sends"] if s["packed"]]
        self.assertTrue(packed)
        self.assertTrue(all(s["saved_bytes"] < 0 for s in packed))
        self.assertTrue(all(s["saved_estimated_text_tokens"] < 0 for s in packed))
        self.assertLess(case["saved_bytes"], 0)
        self.assertLess(case["saved_estimated_text_tokens"], 0)

    def test_report_carries_capture_mode_and_rejects_malformed_control_shapes(self):
        report = evaluate_recordings(_fixture_set(), clock=_deterministic_clock())
        modes = {case["name"]: case["mode"] for case in report["cases"]}
        self.assertEqual(modes["large_tree"], "ax")
        self.assertEqual(modes["multimodal"], "som")

        base = _fixture_set()["cases"][0]
        for control in (
            ["text_payload"],
            {"text_payload": []},
            {"content_blocks": {}},
        ):
            bad = {**base, "control": control}
            with self.subTest(control=control), self.assertRaises(ValueError):
                evaluate_fixture(bad, clock=_deterministic_clock())

        self.assertEqual(observation_eval._missing_fields("not json", "changed"), ["<content>"])

    def test_backwards_clock_never_yields_negative_latency(self):
        ticks = itertools.count(10_000, -100).__next__
        case = evaluate_fixture(_fixture_set()["cases"][0], clock=ticks)
        self.assertTrue(all(send["latency_ns"] == 0 for send in case["sends"]))

    def test_packed_send_reports_no_retained_element_tokens(self):
        fixture = copy.deepcopy(next(x for x in _fixture_set()["cases"] if x["name"] == "large_tree"))
        fixture["capture"]["elements"][0]["element_token"] = "tok_abc"
        fixture["control"]["text_payload"]["note"] = "token tok_abc"
        case = evaluate_fixture(fixture, clock=_deterministic_clock())
        self.assertEqual(case["control_element_tokens"], ["tok_abc"])
        self.assertEqual([s["packed"] for s in case["sends"]], [False, False, True])
        self.assertEqual([s["retained_element_tokens"] for s in case["sends"]], [["tok_abc"], ["tok_abc"], []])

    def test_recall_reassembly_rejects_error_malformed_and_stalled_pages(self):
        for reply, message in (
            ({"error": "gone"}, "gone"),
            ({"content": "no header", "eof": True, "next_offset": 0}, "malformed"),
            ({"content": "h1\nh2\nbody", "eof": False, "next_offset": 0}, "advance"),
            ({"content": "h1\nh2\nbody", "eof": False, "next_offset": None}, "advance"),
            ({"content": "h1\nh2\nbody", "eof": False, "next_offset": "9"}, "advance"),
        ):
            with self.subTest(message=message), mock.patch.object(
                observation_eval, "recall",
                side_effect=[json.dumps(reply), AssertionError("recall looped")],
            ):
                with self.assertRaisesRegex(ValueError, message):
                    observation_eval._recall_exact_text(Path("."), "obs_x")

    def test_recall_page_with_a_single_header_line_is_malformed(self):
        reply = {"content": "h1\nbody", "eof": True, "next_offset": 0}
        with mock.patch.object(observation_eval, "recall", return_value=json.dumps(reply)):
            with self.assertRaisesRegex(ValueError, "malformed"):
                observation_eval._recall_exact_text(Path("."), "obs_x")


class MalformedElementTests(unittest.TestCase):
    def test_non_dict_capture_elements_are_skipped_when_collecting_tokens(self):
        base = _fixture_set()["cases"][0]
        fixture = copy.deepcopy(base)
        fixture["capture"]["elements"] += ["junk", None, 7]
        expected = evaluate_fixture(base, clock=_deterministic_clock())["captured_element_tokens"]
        case = evaluate_fixture(fixture, clock=_deterministic_clock())
        self.assertGreater(expected, 0)
        self.assertEqual(case["captured_element_tokens"], expected)


class ReportShapeTests(unittest.TestCase):
    def _case(self, **over):
        tokens = ["tok_%s" % c for c in "zqmawxbk"]
        case = {
            "name": "n",
            "tool_call_id": "t",
            "capture": {"elements": [{"element_token": t} for t in tokens]},
            "control": {"text_payload": {"tokens": tokens}},
        }
        case.update(over)
        return case, sorted(tokens)

    def test_element_tokens_are_reported_sorted(self):
        case, expected = self._case()
        report = evaluate_fixture(case, clock=_deterministic_clock())
        self.assertEqual(report["control_element_tokens"], expected)
        self.assertEqual(report["sends"][0]["retained_element_tokens"], expected)

    def test_truthy_non_string_name_is_rejected(self):
        case, _ = self._case(name=5)
        with self.assertRaisesRegex(ValueError, "name must be nonempty"):
            evaluate_fixture(case, clock=_deterministic_clock())


if __name__ == "__main__":
    unittest.main()
