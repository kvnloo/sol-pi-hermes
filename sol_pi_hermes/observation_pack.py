"""ObservationPack: archive large tool results; project placeholders at request time.

Port of NVlabs/SoL-Pi observation-pack. The stored conversation is never truncated.
Projection happens on a copy of provider request messages (Hermes
``llm_request`` middleware), not via ``transform_tool_result``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import _hermes_home

THRESHOLD_BYTES = 10 * 1024
FULL_SENDS = 2
PLACEHOLDER_EXCERPT_BYTES = 1024
RECALL_MAX_BYTES = 16 * 1024
RECALL_MAX_LINES = 400
RECALL_HEADER_RESERVE_BYTES = 512
RECALL_HEADER_LINES = 2
OBSERVATION_ID_PATTERN = re.compile(r"^obs_[a-f0-9]{24}$")
EVIDENCE_REDUCER_RECEIPT_PREFIX = "sol_pi_evidence_receipt_v1"
CHARS_PER_TOKEN = 4


def hash_text(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def count_lines(text: str) -> int:
    if not text:
        return 0
    lines = 0 if text.endswith("\n") else 1
    return lines + text.count("\n")


def is_observation_id(obs_id: str) -> bool:
    return bool(OBSERVATION_ID_PATTERN.fullmatch(obs_id))


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def observation_path(runtime_root: Path, obs_id: str) -> Path:
    return runtime_root / "observation-pack" / "objects" / f"{obs_id}.txt"


def ledger_path(runtime_root: Path) -> Path:
    return runtime_root / "observation-pack" / "ledger.jsonl"


def runtime_root_for_session(session_id: str, hermes_home: Path | None = None) -> Path:
    home = hermes_home or _hermes_home()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id) or "default"
    return home / "sol-pi" / safe


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return ""


def is_pure_text_tool_result(message: Mapping[str, Any]) -> bool:
    role = message.get("role")
    if role not in {"tool", "toolResult"}:
        return False
    if message.get("isError") or message.get("is_error"):
        return False
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return True
    if isinstance(content, list) and content and all(
        (isinstance(b, str) or (isinstance(b, Mapping) and b.get("type") == "text"))
        for b in content
    ):
        return True
    return False


def tool_name_of(message: Mapping[str, Any]) -> str:
    return str(message.get("name") or message.get("toolName") or message.get("tool_name") or "tool")


def tool_call_id_of(message: Mapping[str, Any]) -> str:
    return str(
        message.get("tool_call_id")
        or message.get("toolCallId")
        or message.get("id")
        or ""
    )


@dataclass(frozen=True)
class Observation:
    id: str
    content_hash: str
    file_path: Path
    tool_name: str
    text: str
    bytes: int
    lines: int
    tokens: int


def contains_reducer_receipt(text: str) -> bool:
    return any(line == EVIDENCE_REDUCER_RECEIPT_PREFIX for line in text.split("\n"))


def _preserve_serialized_result(text: str) -> bool:
    """Keep failed or ambiguous JSON-looking tool results inline."""
    if not text.lstrip().startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except ValueError:
        # Hermes may append subdirectory hints to JSON. Truncation can also
        # remove its closing delimiter: do not infer success from a parse error.
        return True
    if not isinstance(payload, dict):
        return False
    exit_code = payload.get("exit_code")
    terminal_failure = "output" in payload and type(exit_code) is int and exit_code != 0
    return bool(payload.get("error")) or payload.get("success") is False or terminal_failure


def create_observation(message: Mapping[str, Any], runtime_root: Path) -> Observation | None:
    text = _message_text(message)
    if contains_reducer_receipt(text):
        return None
    size = utf8_len(text)
    if size <= THRESHOLD_BYTES:
        return None
    if _preserve_serialized_result(text):
        return None
    if not str(runtime_root):
        raise ValueError("Persistent SoL-Pi runtime directory is unavailable")
    content_hash = hash_text(text)
    tool_name = tool_name_of(message)
    tool_call_id = tool_call_id_of(message)
    obs_id = f"obs_{hash_text(f'{tool_name}\0{tool_call_id}\0{content_hash}')[:24]}"
    return Observation(
        id=obs_id,
        content_hash=content_hash,
        file_path=observation_path(runtime_root, obs_id),
        tool_name=tool_name,
        text=text,
        bytes=size,
        lines=count_lines(text),
        tokens=estimate_tokens(text),
    )


def ensure_stored(observation: Observation) -> None:
    directory = observation.file_path.parent
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise OSError(f"Observation directory is not a regular directory for {observation.id}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(observation.file_path, flags, 0o600)
        try:
            os.write(fd, observation.text.encode("utf-8"))
        finally:
            os.close(fd)
    except FileExistsError:
        existing = observation.file_path.read_bytes()
        if not observation.file_path.is_file() or observation.file_path.is_symlink():
            raise OSError(f"Content-addressed observation is not a regular file for {observation.id}")
        if len(existing) != observation.bytes:
            raise OSError(f"Content-addressed observation size mismatch for {observation.id}")
        if hash_text(existing) != observation.content_hash:
            raise OSError(f"Content-addressed observation hash mismatch for {observation.id}")


def _complete_line_excerpt(text: str, budget_bytes: int, from_end: bool) -> str:
    lines = re.split(r"(?<=\n)", text)
    if lines and lines[-1] == "":
        lines = lines[:-1]
    selected: list[str] = []
    selected_bytes = 0
    index = len(lines) - 1 if from_end else 0
    while 0 <= index < len(lines):
        line = lines[index]
        line_bytes = utf8_len(line)
        if selected_bytes + line_bytes > budget_bytes:
            break
        if from_end:
            selected.insert(0, line)
        else:
            selected.append(line)
        selected_bytes += line_bytes
        index += -1 if from_end else 1
    return "".join(selected)


def placeholder_for(observation: Observation) -> str:
    head_budget = PLACEHOLDER_EXCERPT_BYTES // 2
    tail_budget = PLACEHOLDER_EXCERPT_BYTES - head_budget
    head = _complete_line_excerpt(observation.text, head_budget, False)
    tail = _complete_line_excerpt(observation.text, tail_budget, True)
    return "\n".join(
        [
            f"[large tool result replaced after its first {FULL_SENDS} provider requests]",
            f"id: {observation.id}",
            f"tool: {observation.tool_name}",
            f"original_bytes: {observation.bytes}",
            f"original_lines: {observation.lines}",
            f"estimated_tokens: {observation.tokens}",
            f'retrieve: call obs_recall with {{"id":"{observation.id}","offset":0}}; continue with returned next_offset',
            f"[first complete lines, up to {head_budget} bytes]",
            head,
            f"[middle omitted; last complete lines, up to {tail_budget} bytes]",
            tail,
            f"[{observation.bytes} original bytes omitted]",
        ]
    )


def append_ledger(runtime_root: Path, entry: Mapping[str, Any]) -> None:
    path = ledger_path(runtime_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(entry)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _clone_message(message: Mapping[str, Any]) -> dict[str, Any]:
    cloned = dict(message)
    content = message.get("content")
    if isinstance(content, list):
        cloned["content"] = [dict(b) if isinstance(b, Mapping) else b for b in content]
    return cloned


def _set_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [{"type": "text", "text": text}]
    else:
        message["content"] = text
    return message


def _prior_assistant_counts(messages: list[Mapping[str, Any]]) -> list[int]:
    counts = [0] * len(messages)
    assistant_count = 0
    for index in range(len(messages) - 1, -1, -1):
        counts[index] = assistant_count
        if messages[index].get("role") == "assistant":
            assistant_count += 1
    return counts


def project_messages(
    messages: Iterable[Mapping[str, Any]],
    runtime_root: Path,
    sent_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Request-time projection. Does not mutate the caller's list or stored history."""
    source = [_clone_message(m) for m in messages]
    counts = sent_counts if sent_counts is not None else {}
    prior = _prior_assistant_counts(source)
    request_index = (prior[0] + 1) if prior else 1
    projected = list(source)

    for index, message in enumerate(source):
        if not is_pure_text_tool_result(message):
            continue
        try:
            observation = create_observation(message, runtime_root)
            if observation is None:
                continue
            ensure_stored(observation)
            send_key = f"{runtime_root}\0{observation.id}"
            previous_sends = counts.get(send_key, prior[index])
            if previous_sends < FULL_SENDS:
                append_ledger(
                    runtime_root,
                    {
                        "event": "full",
                        "id": observation.id,
                        "request": request_index,
                        "tool": observation.tool_name,
                        "originalBytes": observation.bytes,
                    },
                )
                counts[send_key] = previous_sends + 1
                continue
            placeholder = placeholder_for(observation)
            append_ledger(
                runtime_root,
                {
                    "event": "placeholder",
                    "id": observation.id,
                    "request": request_index,
                    "sendNumber": previous_sends + 1,
                    "tool": observation.tool_name,
                    "originalBytes": observation.bytes,
                    "placeholderTokens": estimate_tokens(placeholder),
                },
            )
            projected[index] = _set_text(_clone_message(message), placeholder)
            counts[send_key] = previous_sends + 1
        except Exception:
            # Fail open: packing must never cost the agent its observation.
            continue
    return projected


def read_recall_chunk(
    path: Path,
    offset: int,
    *,
    max_bytes: int,
    max_lines: int,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if offset > size:
        raise ValueError(f"Offset {offset} exceeds observation size {size}")
    with path.open("rb") as handle:
        handle.seek(offset)
        buffer = handle.read(max_bytes + 4)
    end = min(len(buffer), max_bytes)
    newline_count = 0
    for index in range(end):
        if buffer[index] != 0x0A:
            continue
        newline_count += 1
        if newline_count == max_lines:
            end = index + 1
            break
    while end > 0 and end < len(buffer) and (buffer[end] & 0xC0) == 0x80:
        end -= 1
    chunk = buffer[:end]
    next_offset = offset + len(chunk)
    text = chunk.decode("utf-8")
    return {
        "text": text,
        "bytes": len(chunk),
        "lines": count_lines(text) if text else 0,
        "nextOffset": next_offset,
        "eof": next_offset >= size,
    }


def recall(runtime_root: Path, obs_id: str, offset: int = 0) -> str:
    if not is_observation_id(obs_id):
        return json.dumps({"error": f"Unknown observation id: {obs_id}"})
    path = observation_path(runtime_root, obs_id)
    limits = {
        "max_bytes": RECALL_MAX_BYTES - RECALL_HEADER_RESERVE_BYTES,
        "max_lines": RECALL_MAX_LINES - RECALL_HEADER_LINES,
    }
    try:
        chunk = read_recall_chunk(path, offset, **limits)
    except FileNotFoundError:
        return json.dumps({"error": f"Unknown observation id: {obs_id}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    header = (
        f"[obs_recall id={obs_id} offset={offset} next_offset={chunk['nextOffset']} eof={chunk['eof']}]\n"
        f"[chunk_bytes={chunk['bytes']} chunk_lines={chunk['lines']}; use next_offset to continue]"
    )
    content = f"{header}\n{chunk['text']}"
    if utf8_len(content) > RECALL_MAX_BYTES or count_lines(content) > RECALL_MAX_LINES:
        return json.dumps({"error": "Recall output exceeded its hard limit"})
    append_ledger(
        runtime_root,
        {
            "event": "recall",
            "id": obs_id,
            "offset": offset,
            "bytes": chunk["bytes"],
            "lines": chunk["lines"],
            "nextOffset": chunk["nextOffset"],
            "eof": chunk["eof"],
        },
    )
    return json.dumps(
        {
            "id": obs_id,
            "offset": offset,
            "next_offset": chunk["nextOffset"],
            "eof": chunk["eof"],
            "bytes": chunk["bytes"],
            "lines": chunk["lines"],
            "content": content,
        }
    )


def wrap_select_context(engine: Any, runtime_root: Path, sent_counts: dict[str, int]) -> None:
    """Compose ObservationPack onto an existing engine without replacing it.

    Hermes skips the ABC default ``select_context``. Binding an instance method
    enables request-time projection while leaving ``compress`` / ``should_compress``
    on the live engine.
    """
    original = getattr(engine, "select_context", None)

    def select_context(request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0, **kwargs):
        selected = None
        if callable(original):
            try:
                selected = original(
                    request_messages,
                    conversation_messages=conversation_messages,
                    incoming_message=incoming_message,
                    budget_tokens=budget_tokens,
                    **kwargs,
                )
            except TypeError:
                selected = original(request_messages)
        base = selected if isinstance(selected, list) and selected else request_messages
        return project_messages(base, runtime_root, sent_counts)

    engine.select_context = select_context  # instance attr; Hermes calls without injecting self
