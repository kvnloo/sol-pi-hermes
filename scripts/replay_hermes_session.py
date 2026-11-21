"""Offline ObservationPack replay over transcript prefixes in a copied Hermes SQLite DB."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sol_pi_hermes.observation_pack import (  # noqa: E402
    estimate_tokens,
    project_request,
    recall,
)

MAX_ACTIVE_ROWS = 2000
CONTENT_JSON_PREFIX = "\0json:"
USAGE_FIELDS = (
    "api_call_count", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
)
MESSAGE_FIELDS = (
    "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
    "tool_name", "active", "api_content",
)
PLACEHOLDER_ID = re.compile(r"\A\[large tool result [^\n]*\]\nid: (obs_[a-f0-9]{24})\n")


class ReplayError(ValueError):
    """Input cannot be used for this narrow replay."""


def require_tables(conn):
    for table, required in (
        ("sessions", ("id", "parent_session_id", *USAGE_FIELDS)),
        ("messages", MESSAGE_FIELDS),
    ):
        # Require actual tables, not views, before selecting user data.
        kind = conn.execute(
            "SELECT type FROM main.sqlite_schema WHERE name = ?", (table,)
        ).fetchone()
        columns = {r["name"] for r in conn.execute(f"PRAGMA main.table_info({table})")}
        if kind is None or kind["type"] != "table" or not set(required) <= columns:
            raise ReplayError(f"unsupported Hermes SQLite schema: {table}")


def message_from_row(row):
    role = row["role"]
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    content = row["content"]
    if isinstance(content, str) and content.startswith(CONTENT_JSON_PREFIX):
        try:
            content = json.loads(content[len(CONTENT_JSON_PREFIX):])
        except ValueError:
            # Matches the upstream decoder's literal fallback.
            pass
    sidecar = row["api_content"]
    if role in {"user", "assistant"} and isinstance(sidecar, str) and sidecar:
        content = sidecar
    if content is not None and not isinstance(content, (str, list, dict)):
        raise ReplayError("unsupported message content type")
    message = {"role": role, "content": content}
    if role == "tool":
        for key in ("tool_name", "tool_call_id"):
            if isinstance(row[key], str) and row[key]:
                message[key] = row[key]
    if role == "assistant" and row["tool_calls"]:
        try:
            calls = json.loads(row["tool_calls"])
        except (TypeError, ValueError):
            raise ReplayError("unsupported assistant tool_calls encoding") from None
        if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
            raise ReplayError("unsupported assistant tool_calls structure")
        message["tool_calls"] = calls
    return message


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else part["text"]
            for part in content
            if isinstance(part, str) or (
                isinstance(part, dict) and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        )
    return ""


def measure_messages(messages):
    encoded = json.dumps(messages, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return len(encoded), sum(estimate_tokens(content_text(m["content"])) for m in messages)


def recall_metrics(root, obs_id, original):
    """Measure a single complete recall sweep for one projected archive object."""
    offset = pages = wire_bytes = largest_page = 0
    body = original.encode("utf-8")
    matches = True
    while True:
        wire = recall(root, obs_id, offset)
        pages += 1
        page_bytes = len(wire.encode("utf-8"))
        wire_bytes += page_bytes
        largest_page = max(largest_page, page_bytes)
        page = json.loads(wire)
        if "error" in page:
            return {"observation_id": obs_id, "page_count": pages, "json_bytes": wire_bytes,
                    "largest_page_json_bytes": largest_page,
                    "byte_exact": False, "status": "recall_error"}
        part = page["content"].split("\n", 2)
        end = page["next_offset"]
        if len(part) != 3 or type(end) is not int or not offset < end <= len(body):
            return {"observation_id": obs_id, "page_count": pages, "json_bytes": wire_bytes,
                    "largest_page_json_bytes": largest_page,
                    "byte_exact": False, "status": "invalid_page"}
        matches &= part[2].encode("utf-8") == body[offset:end]
        offset = end
        if page["eof"]:
            exact = matches and offset == len(body)
            return {"observation_id": obs_id, "page_count": pages, "json_bytes": wire_bytes,
                    "largest_page_json_bytes": largest_page,
                    "byte_exact": exact, "status": "complete" if exact else "mismatch"}


def evaluate_session(conn, session_id, root):
    session = conn.execute(
        f"SELECT parent_session_id, {', '.join(USAGE_FIELDS)} FROM main.sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if session is None:
        raise ReplayError("requested session not found")
    rows = conn.execute(
        "SELECT id, role, content, tool_call_id, tool_calls, tool_name, api_content "
        "FROM main.messages WHERE session_id = ? AND active = 1 ORDER BY id LIMIT ?",
        (session_id, MAX_ACTIVE_ROWS + 1),
    ).fetchall()
    if len(rows) > MAX_ACTIVE_ROWS:
        raise ReplayError(f"session exceeds {MAX_ACTIVE_ROWS} active rows")
    prefix, boundaries, recalls = [], [], {}
    skipped_roles = 0
    sent_counts = {}
    for row in rows:
        msg = message_from_row(row)
        if msg is None:
            skipped_roles += 1
            continue
        # Model only durable response boundaries; pending tail, retries and
        # unrecorded calls cannot be established from the transcript.
        if msg["role"] == "assistant" and prefix and prefix[-1]["role"] in {"user", "tool"}:
            result = project_request({"messages": prefix}, root, sent_counts)
            projected = result["request"]["messages"] if result else prefix
            before_bytes, before_tokens = measure_messages(prefix)
            after_bytes, after_tokens = measure_messages(projected)
            placeholders = 0
            for before, after in zip(prefix, projected, strict=True):
                if before == after:
                    continue
                match = PLACEHOLDER_ID.match(content_text(after["content"]))
                if match is None:
                    raise ReplayError("unexpected ObservationPack projection")
                placeholders += 1
                obs_id = match[1]
                if obs_id not in recalls:
                    recalls[obs_id] = recall_metrics(root, obs_id, content_text(before["content"]))
            boundaries.append({
                "assistant_row_id": row["id"],
                "before_message_json_bytes": before_bytes, "after_message_json_bytes": after_bytes,
                "before_estimated_content_text_tokens": before_tokens,
                "after_estimated_content_text_tokens": after_tokens,
                "placeholder_count": placeholders,
            })
        prefix.append(msg)

    totals = {field: sum(row[field] for row in boundaries) for field in (
        "before_message_json_bytes", "after_message_json_bytes",
        "before_estimated_content_text_tokens", "after_estimated_content_text_tokens",
        "placeholder_count",
    )}
    totals["saved_message_json_bytes"] = totals["before_message_json_bytes"] - totals["after_message_json_bytes"]
    totals["saved_estimated_content_text_tokens"] = (
        totals["before_estimated_content_text_tokens"] - totals["after_estimated_content_text_tokens"]
    )
    return {
        "session_id": session_id, "active_rows": len(rows), "skipped_role_rows": skipped_roles,
        "has_parent_not_replayed": session["parent_session_id"] is not None,
        "observed_session_usage": {field: session[field] for field in USAGE_FIELDS},
        "actual_projection_token_delta": None,
        "inferred_request_opportunities": len(boundaries), "boundaries": boundaries,
        "totals": totals, "recall": list(recalls.values()),
    }


def replay(db_path, session_ids):
    # Open an operator-supplied snapshot, never discover a live Hermes home.
    if not db_path.is_file():
        raise ReplayError("SQLite snapshot file not found")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA trusted_schema = OFF")
        conn.execute("BEGIN")
        require_tables(conn)
        with tempfile.TemporaryDirectory(prefix="sol-pi-session-replay-") as temp:
            sessions = [
                evaluate_session(conn, session_id, Path(temp) / str(index))
                for index, session_id in enumerate(session_ids)
            ]
    return {
        "schema_version": 1,
        "mode": "offline durable active-row message-prefix approximation",
        "metrics": {
            "message_json_bytes": "compact ASCII-escaped JSON encoding of the reconstructed message array, not wire bytes",
            "estimated_content_text_tokens": "sum of ceil(text characters/4) by message, not tokenizer/provider usage",
            "observed_session_usage": "stored historical session counters, not per-boundary or counterfactual",
            "recall": "one complete simulated recall sweep per distinct projected observation, not observed tool calls",
        },
        "sessions": sessions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="operator-created offline SQLite backup, not the live database")
    parser.add_argument("session_ids", nargs="+", help="exact session IDs to replay")
    args = parser.parse_args(argv)
    if len(set(args.session_ids)) != len(args.session_ids):
        parser.error("session IDs must be distinct")
    try:
        report = replay(args.snapshot, args.session_ids)
    except ReplayError as exc:
        parser.error(str(exc))
    except (OSError, sqlite3.Error, UnicodeError):
        parser.error("failed to read or evaluate SQLite snapshot")
    print(json.dumps(report, indent=2))
    return int(any(not recall_row["byte_exact"] for s in report["sessions"] for recall_row in s["recall"]))


if __name__ == "__main__":
    raise SystemExit(main())
