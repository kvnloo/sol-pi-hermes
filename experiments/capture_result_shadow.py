"""Shadow evaluator for Hermes CaptureResult-shaped AX/SOM text observations.

This does not import Hermes and does not alter provider-visible behavior. It
builds the stable text/JSON portion of the current CaptureResult contract, then
runs SoL-Pi ObservationPack as a shadow projection over frozen synthetic
CaptureResult-shaped fixtures.

Images are intentionally out of scope: ObservationPack currently operates on
large pure-text tool results, not multimodal screenshot blocks.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sol_pi_hermes.observation_pack import (
    OBSERVATION_ID_PATTERN,
    estimate_tokens,
    project_messages,
    recall,
    utf8_len,
)

SCHEMA = "sol_pi_hermes.capture_shadow.v1"


@dataclass(frozen=True)
class Element:
    index: int
    role: str
    label: str
    bounds: tuple[int, int, int, int]
    app: str = "FixtureApp"
    pid: int = 100
    window_id: int = 7
    element_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "index": self.index,
            "role": self.role,
            "label": self.label,
            "bounds": list(self.bounds),
            "app": self.app,
            "pid": self.pid,
            "window_id": self.window_id,
        }
        if self.element_token:
            out["element_token"] = self.element_token
        return out


@dataclass(frozen=True)
class CaptureFixture:
    id: str
    elements: tuple[Element, ...]
    required_markers: tuple[str, ...]
    mode: str = "ax"
    width: int = 1440
    height: int = 900
    app: str = "FixtureApp"
    window_title: str = "Fixture"
    ax_max_elements: int = 0
    note: str = ""

    def provider_text(self) -> str:
        visible = [e.to_dict() for e in self.elements]
        lines = [
            f"capture mode={self.mode} {self.width}x{self.height} app={self.app} window={self.window_title!r}",
            f"{len(visible)} interactable element(s):",
        ]
        for e in visible:
            token = f" token={e.get('element_token')}" if e.get("element_token") else ""
            lines.append(
                f"[{e['index']}] {e['role']} label={e['label']!r} bounds={e['bounds']}{token}"
            )
        if self.note:
            lines.append(f"({self.note})")
        return json.dumps(
            {
                "mode": self.mode,
                "width": self.width,
                "height": self.height,
                "app": self.app,
                "window_title": self.window_title,
                "elements": visible,
                "total_elements": len(visible),
                "summary": "\n".join(lines),
                **({"ax_max_elements": self.ax_max_elements} if self.ax_max_elements else {}),
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )


def _filler_elements(n: int, *, label_size: int = 80) -> list[Element]:
    return [
        Element(
            index=i + 1,
            role="AXStaticText" if i % 4 else "AXButton",
            label=f"routine-{i:03d}-" + ("x" * label_size),
            bounds=((i * 17) % 1200, (i * 11) % 780, 120, 24),
            element_token=f"tok-{i:03d}",
        )
        for i in range(n)
    ]


def default_fixtures() -> tuple[CaptureFixture, ...]:
    large = _filler_elements(180)
    large[90] = Element(
        index=91,
        role="AXButton",
        label="CRITICAL_MID_ACTION",
        bounds=(700, 440, 140, 28),
        element_token="critical-mid-token",
    )

    modal = _filler_elements(150)
    modal[0] = Element(
        index=1,
        role="AXDialog",
        label="MODAL_BLOCKER_CONFIRM_DELETE",
        bounds=(420, 250, 600, 300),
        element_token="modal-root",
    )

    ambiguous = _filler_elements(170)
    ambiguous[78] = Element(79, "AXButton", "SAVE_PRIMARY", (900, 700, 90, 28), element_token="save-primary")
    ambiguous[79] = Element(80, "AXButton", "SAVE_COPY", (1000, 700, 90, 28), element_token="save-copy")

    stale = _filler_elements(165)
    stale[82] = Element(
        83,
        "AXButton",
        "STALE_TARGET",
        (500, 300, 120, 28),
        element_token="stale-token-v1",
    )
    stale[83] = Element(
        84,
        "AXButton",
        "CURRENT_TARGET",
        (500, 340, 120, 28),
        element_token="current-token-v2",
    )

    repeated = _filler_elements(160)
    repeated[-1] = Element(
        160,
        "AXButton",
        "REPEATED_CAPTURE_SENTINEL",
        (1100, 760, 180, 28),
        element_token="repeat-token",
    )

    return (
        CaptureFixture("large_tree", tuple(large), ("CRITICAL_MID_ACTION", "critical-mid-token")),
        CaptureFixture("modal", tuple(modal), ("MODAL_BLOCKER_CONFIRM_DELETE", "modal-root")),
        CaptureFixture("ambiguous", tuple(ambiguous), ("SAVE_PRIMARY", "SAVE_COPY")),
        CaptureFixture("stale_tokens", tuple(stale), ("STALE_TARGET", "CURRENT_TARGET", "current-token-v2")),
        CaptureFixture("repeated", tuple(repeated), ("REPEATED_CAPTURE_SENTINEL", "repeat-token")),
    )


def _obs_id(text: str) -> str:
    match = re.search(r"\bid:\s*(obs_[a-f0-9]{24})\b", text)
    if not match or not OBSERVATION_ID_PATTERN.fullmatch(match.group(1)):
        raise ValueError("ObservationPack placeholder missing observation id")
    return match.group(1)


def _recall_all(root: Path, obs_id: str) -> tuple[str, int]:
    offset = 0
    chunks: list[str] = []
    calls = 0
    while True:
        row = json.loads(recall(root, obs_id, offset))
        if row.get("error"):
            raise ValueError(str(row["error"]))
        # Recall has two display-header lines. Byte offsets address only the
        # archived body; inserting headers/newlines between pages corrupts it.
        chunks.append(str(row["content"]).split("\n", 2)[2])
        calls += 1
        if row.get("eof"):
            break
        next_offset = int(row["next_offset"])
        if next_offset <= offset:
            raise RuntimeError("recall made no progress")
        offset = next_offset
        if calls > 128:
            raise RuntimeError("recall exceeded safety bound")
    return "".join(chunks), calls


def evaluate_fixture(fixture: CaptureFixture, root: Path) -> dict[str, Any]:
    text = fixture.provider_text()
    if utf8_len(text) <= 10 * 1024:
        raise ValueError(f"fixture {fixture.id} does not cross ObservationPack threshold")

    message = {
        "role": "tool",
        "name": "computer_use",
        "tool_call_id": f"capture-{fixture.id}",
        "content": text,
    }
    counts: dict[str, int] = {}

    t0 = time.perf_counter()
    first = project_messages([message], root, counts)[0]
    second = project_messages([message], root, counts)[0]
    third = project_messages([message], root, counts)[0]
    projection_ms = (time.perf_counter() - t0) * 1000.0

    first_text = str(first["content"])
    second_text = str(second["content"])
    placeholder = str(third["content"])
    if first_text != text or second_text != text:
        raise AssertionError("ObservationPack changed one of the first two sends")
    if placeholder == text:
        raise AssertionError("third send was not projected")

    obs_id = _obs_id(placeholder)
    recalled, recall_calls = _recall_all(root, obs_id)

    required = list(fixture.required_markers)
    placeholder_retained = [m for m in required if m in placeholder]
    recall_recovered = [m for m in required if m in recalled]

    original_bytes = utf8_len(text)
    placeholder_bytes = utf8_len(placeholder)
    return {
        "schema": SCHEMA,
        "fixture": fixture.id,
        "original_bytes": original_bytes,
        "placeholder_bytes": placeholder_bytes,
        "byte_reduction": original_bytes - placeholder_bytes,
        "compression_ratio": placeholder_bytes / original_bytes,
        "original_estimated_tokens": estimate_tokens(text),
        "placeholder_estimated_tokens": estimate_tokens(placeholder),
        "projection_ms_3_requests": projection_ms,
        "required_markers": required,
        "placeholder_retained": placeholder_retained,
        "placeholder_retention_rate": len(placeholder_retained) / len(required),
        "recall_recovered": recall_recovered,
        "recall_recovery_rate": len(recall_recovered) / len(required),
        "recall_exact": recalled.encode("utf-8") == text.encode("utf-8"),
        "recall_calls": recall_calls,
        "observation_id": obs_id,
        "full_send_1_unchanged": first_text == text,
        "full_send_2_unchanged": second_text == text,
    }


def run(fixtures: Iterable[CaptureFixture] | None = None, *, root: Path | None = None) -> dict[str, Any]:
    fixtures = tuple(fixtures or default_fixtures())
    if root is not None:
        rows = [evaluate_fixture(f, root / f.id) for f in fixtures]
    else:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = [evaluate_fixture(f, base / f.id) for f in fixtures]
    return {
        "schema": SCHEMA,
        "mode": "shadow",
        "scope": "Hermes CaptureResult-shaped AX/SOM pure-text projection; screenshots excluded",
        "rows": rows,
        "all_first_two_unchanged": all(
            r["full_send_1_unchanged"] and r["full_send_2_unchanged"] for r in rows
        ),
        "all_markers_recoverable": all(r["recall_recovery_rate"] == 1.0 for r in rows),
        "all_observations_exact": all(r["recall_exact"] for r in rows),
        "placeholder_retention_is_not_equivalence": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
