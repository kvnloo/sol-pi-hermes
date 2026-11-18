"""Action Fusion: mutate a file then run a follow-up command in one observation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

THEN_RUN_SUCCEEDED = "[then_run:succeeded]"
THEN_RUN_UNKNOWN = "[then_run:unknown]"
THEN_RUN_FAILED = "[then_run:failed]"
THEN_RUN_SKIPPED = "[then_run:skipped]"

THEN_RUN_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command without NUL characters to run after the mutation succeeds"},
        "timeout": {"type": "number", "minimum": 0, "description": "Finite non-negative timeout in seconds (optional; 0 uses the host default)"},
    },
    "required": ["command"],
}

WRITE_SCHEMA = {
    "name": "sol_pi_write",
    "description": (
        "Write a file then optionally run a follow-up command in the same observation "
        "(Action Fusion). Use instead of write_file + terminal when the next step is "
        "run/test/build of that file. Skipped if the write fails."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "File contents"},
            "then_run": THEN_RUN_SCHEMA,
        },
        "required": ["path", "content"],
    },
}

PATCH_SCHEMA = {
    "name": "sol_pi_patch",
    "description": (
        "Patch a file then optionally run a follow-up command in the same observation "
        "(Action Fusion). Use instead of patch + terminal for the next run/test/build."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to patch"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "then_run": THEN_RUN_SCHEMA,
        },
        "required": ["path"],
    },
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_unchanged_before_command(path: Path, mutation_hash: str) -> None:
    if not path.is_file():
        raise OSError(f"{THEN_RUN_SKIPPED} target missing; the command was not run.")
    if file_sha256(path) != mutation_hash:
        raise OSError(f"{THEN_RUN_SKIPPED} target content changed after the fused mutation; the command was not run.")


def _reported_failure(result: str) -> str | None:
    """Hermes tool dispatch reports failures as JSON, not only exceptions."""
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        return str(payload["error"])
    if payload.get("success") is False:
        return str(payload.get("message") or "Tool reported an unsuccessful result")
    exit_code = payload.get("exit_code")
    if type(exit_code) is int and exit_code != 0:
        return f"Command exited with code {exit_code}"
    return None


def execute_mutation_then_run(
    *,
    mutate: Callable[[], str],
    absolute_path: Path | None,
    then_run: dict[str, Any] | None,
    run_command: Callable[[str, float | None], str],
) -> str:
    if then_run is not None:
        valid = isinstance(then_run, dict) and isinstance(then_run.get("command"), str) and bool(then_run["command"].strip()) and "\0" not in then_run["command"]
        if valid and then_run.get("timeout") is not None:
            timeout = then_run["timeout"]
            try:
                valid = type(timeout) in (int, float) and timeout >= 0 and math.isfinite(timeout)
            except OverflowError:
                valid = False
        if not valid:
            return json.dumps({"error": "Invalid then_run: require a nonempty NUL-free string command and finite non-negative numeric timeout; no mutation was attempted.",
                               "then_run": THEN_RUN_SKIPPED})
    try:
        mutation_result = mutate()
    except Exception as exc:
        if then_run is not None:
            return json.dumps(
                {
                    "error": f"{exc}\n\n{THEN_RUN_SKIPPED} The file mutation did not complete successfully; the command was not run."
                }
            )
        return json.dumps({"error": str(exc)})

    if not then_run:
        return mutation_result if isinstance(mutation_result, str) else json.dumps({"result": mutation_result})

    failure = _reported_failure(mutation_result)
    try:
        payload = json.loads(mutation_result)
    except (TypeError, ValueError):
        payload = None
    # Hermes PatchResult reports success; WriteResult reports bytes_written.
    # Absence of an error alone is not proof that a mutation completed.
    confirmed = isinstance(payload, dict) and (
        payload.get("success") is True
        or ("success" not in payload and type(payload.get("bytes_written")) is int
            and payload["bytes_written"] >= 0)
    )
    if failure or not confirmed:
        return json.dumps({"mutation": mutation_result, "then_run": THEN_RUN_SKIPPED,
                           "error": failure or "File mutation success was not confirmed; the command was not run."})

    try:
        # Only callers with a verified host-local target may enable this guard.
        # Backend paths (even absolute ones) are not necessarily host paths.
        if absolute_path is not None:
            mutation_hash = file_sha256(absolute_path)
            assert_unchanged_before_command(absolute_path, mutation_hash)
        output = run_command(str(then_run.get("command") or ""), then_run.get("timeout"))
        failure = _reported_failure(output)
        status = THEN_RUN_FAILED if failure else THEN_RUN_SUCCEEDED
        try:
            payload = json.loads(output)
        except (TypeError, ValueError):
            payload = None
        if not failure and not (isinstance(payload, dict) and type(payload.get("exit_code")) is int):
            status = THEN_RUN_UNKNOWN
        return json.dumps(
            {
                "mutation": mutation_result,
                "then_run": status,
                "output": output,
            }
        )
    except Exception as exc:
        return json.dumps(
            {
                "mutation": mutation_result,
                "then_run": THEN_RUN_FAILED,
                "error": str(exc),
            }
        )
