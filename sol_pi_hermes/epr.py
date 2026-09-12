"""Evidence-Preserving Reducer: quote-verify a diagnostic log; fail open.

Uses ``transform_tool_result`` because the stored tool result *is* the
verified receipt (SoL-Pi's tool_result seam). ObservationPack later skips
receipts so they are not packed again.
"""

from __future__ import annotations

import json
import re
from typing import Any

DIAGNOSTIC_COMMAND = re.compile(
    r"(?:^|[;&|()\s])(?:lake\s+build|lake\s+env\s+lean|lean|coq|"
    r"cargo(?:\s+(?:build|test|check))?|zig\s+build|pytest|"
    r"python(?:3)?\s+-m\s+(?:pytest|unittest|py_compile)|ctest|"
    r"cmake\s+--build|ninja|make|npm\s+test|pnpm\s+test|yarn\s+test|"
    r"go\s+test|bazel\s+test)(?:\s|$)",
    re.IGNORECASE,
)
LIKELY_SECRET = re.compile(
    r"(?:api[_-]?key|authorization|bearer|access[_-]?token|secret)[^\n]{0,32}[=:][^\n]+",
    re.IGNORECASE,
)
RECEIPT_PREFIX = "sol_pi_evidence_receipt_v1"
MIN_BYTES = 4096


def looks_like_diagnostic(command: str) -> bool:
    return bool(DIAGNOSTIC_COMMAND.search(command or ""))


def should_reduce(body: str, command: str) -> bool:
    if not looks_like_diagnostic(command):
        return False
    if len(body.encode("utf-8")) < MIN_BYTES:
        return False
    if LIKELY_SECRET.search(body):
        return False
    return True


def quote_verify(receipt_quotes: list[str], archive_body: str) -> bool:
    """Every quoted line must appear byte-for-byte in the archive."""
    if not receipt_quotes:
        return False
    return all(quote in archive_body for quote in receipt_quotes)


def apply_receipt(command: str, archive_hash: str, quotes: list[str], notes: str) -> str:
    lines = [
        RECEIPT_PREFIX,
        f"command: {command}",
        f"source_sha256: {archive_hash}",
        "quotes:",
        *[f"- {q}" for q in quotes],
        "notes:",
        notes,
    ]
    return "\n".join(lines)


def transform_if_reduced(
    *,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    reducer: Any | None,
) -> str | None:
    """Return a replacement string or None to keep the original (fail open)."""
    if tool_name not in {"terminal", "bash", "execute"}:
        return None
    command = str(args.get("command") or args.get("cmd") or "")
    if not should_reduce(result, command):
        return None
    if reducer is None:
        return None
    try:
        reduced = reducer(command=command, body=result)
    except Exception:
        return None
    if not isinstance(reduced, dict):
        return None
    quotes = reduced.get("quotes") or []
    if not isinstance(quotes, list) or not quote_verify([str(q) for q in quotes], result):
        return None
    from .observation_pack import hash_text

    receipt = apply_receipt(command, hash_text(result), [str(q) for q in quotes], str(reduced.get("notes") or ""))
    if len(receipt.encode("utf-8")) >= len(result.encode("utf-8")):
        return None
    return receipt
