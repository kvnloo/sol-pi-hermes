"""Offline, read-only ObservationPack shadow replay of frozen Hermes CaptureResult fixtures.

The control is the recorded provider-visible capture payload; the candidate is
the plugin's actual request-time projection. No computer-use action or model
request is made. Only the temporary ObservationPack archive is written.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .observation_pack import FULL_SENDS, estimate_tokens, project_request, recall, utf8_len

DEFAULT_FIXTURES = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "observation_pack_capture_results.json"
)
_OBSERVATION_ID = re.compile(r"(?m)^id: (obs_[a-f0-9]{24})$")
_ELEMENT_INDEX = re.compile(r'"index"\s*:\s*(\d+)')


def _control_content(fixture: Mapping[str, Any]) -> str | list[Any]:
    """Rehydrate a frozen Hermes JSON text result or native multimodal blocks."""
    control = fixture["control"]
    if not isinstance(control, dict):
        raise ValueError("control must be a JSON object")
    if ("text_payload" in control) == ("content_blocks" in control):
        raise ValueError("control needs exactly one of text_payload or content_blocks")
    if "text_payload" in control:
        if not isinstance(control["text_payload"], dict):
            raise ValueError("text_payload must be a JSON object")
        # Hermes computer_use.tool._text_capture_payload uses json.dumps defaults.
        return json.dumps(control["text_payload"])
    blocks = control["content_blocks"]
    if not isinstance(blocks, list):
        raise ValueError("content_blocks must be a list")
    return copy.deepcopy(blocks)


def _text_part(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in content
        if isinstance(block, str)
        or (isinstance(block, dict) and block.get("type") == "text")
    )


def _bytes(content: str | list[Any]) -> int:
    if isinstance(content, str):
        return utf8_len(content)
    # The multimodal wire-size proxy includes the image URL/base64; its model
    # token cost is deliberately NOT inferred from its byte length.
    return utf8_len(json.dumps(content, ensure_ascii=False, sort_keys=True))


def _indices(text: str) -> list[int]:
    return sorted({int(match) for match in _ELEMENT_INDEX.findall(text)})


def _tokens(text: str, raw_tokens: set[str]) -> list[str]:
    return sorted(token for token in raw_tokens if token in text)


def _missing_fields(original: str, candidate: str) -> list[str]:
    """Top-level JSON fields no longer available as structured visible data."""
    if original == candidate:
        return []
    try:
        before = json.loads(original)
        after = json.loads(candidate)
    except (ValueError, TypeError):
        try:
            before = json.loads(original)
        except (ValueError, TypeError):
            return ["<content>"]
        return sorted(before) if isinstance(before, dict) else ["<content>"]
    if not isinstance(before, dict) or not isinstance(after, dict):
        return ["<content>"]
    return sorted(key for key, value in before.items() if key not in after or after[key] != value)


def _recall_exact_text(root: Path, observation_id: str) -> str:
    """Reassemble every obs_recall page, stripping its two-line response header."""
    offset = 0
    parts: list[str] = []
    while True:
        payload = json.loads(recall(root, observation_id, offset))
        if "error" in payload:
            raise ValueError(payload["error"])
        header_and_text = payload["content"].split("\n", 2)
        if len(header_and_text) != 3:
            raise ValueError("obs_recall returned a malformed page")
        parts.append(header_and_text[2])
        next_offset = payload["next_offset"]
        if payload["eof"]:
            return "".join(parts)
        if not isinstance(next_offset, int) or next_offset <= offset:
            raise ValueError("obs_recall did not advance its byte offset")
        offset = next_offset


def evaluate_fixture(
    fixture: Mapping[str, Any],
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Compare three provider sends; first two must remain the original control."""
    absent = [k for k in ("name", "capture", "control", "tool_call_id") if k not in fixture]
    if absent:
        raise ValueError(f"fixture missing required field(s): {', '.join(absent)}")
    capture = fixture["capture"]
    if not isinstance(capture, dict) or not isinstance(capture.get("elements"), list):
        raise ValueError("capture must be a CaptureResult-shaped object with elements")
    name = fixture["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("fixture name must be nonempty")
    original_content = _control_content(fixture)
    raw_tokens = {
        element["element_token"]
        for element in capture["elements"]
        if isinstance(element, dict)
        and isinstance(element.get("element_token"), str)
        and element["element_token"]
    }
    message = {
        "role": "tool",
        "name": "computer_use",
        "tool_call_id": str(fixture["tool_call_id"]),
        "content": original_content,
    }
    request = {"model": "offline-shadow-only", "messages": [message]}
    frozen = copy.deepcopy(request)
    # Text recall cannot certify image, metadata or malformed-block preservation.
    # Keep this control-side rule independent of the candidate projection gate.
    requires_passthrough = isinstance(original_content, list) and any(
        not (
            isinstance(block, str)
            or (isinstance(block, dict) and set(block) == {"type", "text"}
                and block.get("type") == "text" and isinstance(block.get("text"), str))
        )
        for block in original_content
    )
    original_text = _text_part(original_content)
    original_bytes = _bytes(original_content)
    original_estimated_tokens = estimate_tokens(original_text)
    sent_counts: dict[str, int] = {}
    sends = []

    with tempfile.TemporaryDirectory(prefix="sol-pi-shadow-") as tmp:
        root = Path(tmp)
        for number in range(1, FULL_SENDS + 2):
            start_ns = clock()
            projected = project_request(request, root, sent_counts)
            latency_ns = max(0, clock() - start_ns)
            # Preserve the original model-visible request as the control.
            visible = (projected["request"] if projected is not None else request)["messages"][0]["content"]
            visible_text = _text_part(visible)
            packed = visible != original_content
            obs_id = None
            exact_recall = None
            lost_after_recall: list[str] = []
            if packed:
                found = _OBSERVATION_ID.search(visible_text)
                if not found:
                    raise ValueError(f"{name}: changed capture lacks an observation id")
                obs_id = found.group(1)
                reconstructed = _recall_exact_text(root, obs_id)
                exact_recall = reconstructed == original_text
                lost_after_recall = _missing_fields(original_text, reconstructed)

            after_bytes = _bytes(visible)
            after_estimated_tokens = estimate_tokens(visible_text)
            sends.append(
                {
                    "send": number,
                    "packed": packed,
                    "observation_id": obs_id,
                    "before_bytes": original_bytes,
                    "after_bytes": after_bytes,
                    "saved_bytes": original_bytes - after_bytes,
                    "before_estimated_text_tokens": original_estimated_tokens,
                    "after_estimated_text_tokens": after_estimated_tokens,
                    "saved_estimated_text_tokens": original_estimated_tokens - after_estimated_tokens,
                    "retained_indices": _indices(visible_text),
                    "retained_element_tokens": _tokens(visible_text, raw_tokens),
                    "visible_missing_fields": (
                        _missing_fields(original_text, visible_text)
                        if isinstance(original_content, str) and isinstance(visible, str)
                        else []
                    ),
                    "byte_exact_recall": exact_recall,
                    "lost_fields_after_recall": lost_after_recall,
                    "latency_ns": latency_ns,
                }
            )
    return {
        "name": name,
        "mode": capture.get("mode"),
        "captured_element_count": len(capture["elements"]),
        "captured_element_tokens": len(raw_tokens),
        "control_indices": _indices(original_text),
        "control_element_tokens": _tokens(original_text, raw_tokens),
        "control_unchanged": request == frozen,
        "non_text_content_unchanged": not requires_passthrough or all(
            not send["packed"] for send in sends
        ),
        "sends": sends,
        "saved_bytes": sum(item["saved_bytes"] for item in sends),
        "saved_estimated_text_tokens": sum(item["saved_estimated_text_tokens"] for item in sends),
    }


def evaluate_recordings(
    recordings: Mapping[str, Any],
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    if recordings.get("schema_version") != 1 or not isinstance(recordings.get("cases"), list):
        raise ValueError("expected schema_version=1 and cases array")
    missing = [key for key in ("host_main_sha", "provenance") if key not in recordings]
    if missing:
        raise ValueError(f"recordings missing required field(s): {', '.join(missing)}")
    if not recordings["cases"]:
        raise ValueError("cases must contain at least one fixture")
    cases = [evaluate_fixture(case, clock=clock) for case in recordings["cases"]]
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")
    rows = [send for case in cases for send in case["sends"]]
    before = sum(row["before_bytes"] for row in rows)
    saved = sum(row["saved_bytes"] for row in rows)
    return {
        "schema_version": 1,
        "host_main_sha": recordings["host_main_sha"],
        "fixture_provenance": recordings["provenance"],
        "method": "request-time shadow only; original provider-visible control retained",
        "token_estimator": "ceil(text characters / 4); excludes image tokens and provider/cache billing",
        "latency": "local measured wall time in ns; timings vary between runs",
        "cases": cases,
        "totals": {
            "before_bytes": before,
            "after_bytes": before - saved,
            "saved_bytes": saved,
            "saved_percent": round(100 * saved / before, 2) if before else 0.0,
            "saved_estimated_text_tokens": sum(row["saved_estimated_text_tokens"] for row in rows),
            "projection_latency_ns": sum(row["latency_ns"] for row in rows),
            "lost_fields_after_recall": sorted(
                {field for row in rows for field in row["lost_fields_after_recall"]}
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    recordings = json.loads(args.fixtures.read_text(encoding="utf-8"))
    report = evaluate_recordings(recordings)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return int(
        bool(report["totals"]["lost_fields_after_recall"])
        or any(not case["control_unchanged"] for case in report["cases"])
        or any(not case["non_text_content_unchanged"] for case in report["cases"])
        or any(
            send["packed"] and send["send"] <= FULL_SENDS
            for case in report["cases"]
            for send in case["sends"]
        )
        or any(send["byte_exact_recall"] is False for case in report["cases"] for send in case["sends"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
