"""Action Fusion: mutate a file then run a follow-up command in one observation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

THEN_RUN_SUCCEEDED = "[then_run:succeeded]"
THEN_RUN_FAILED = "[then_run:failed]"
THEN_RUN_SKIPPED = "[then_run:skipped]"

THEN_RUN_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to run after the mutation succeeds"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
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


def execute_mutation_then_run(
    *,
    mutate: Callable[[], str],
    absolute_path: Path,
    then_run: dict[str, Any] | None,
    run_command: Callable[[str, float | None], str],
) -> str:
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

    try:
        mutation_hash = file_sha256(absolute_path)
        assert_unchanged_before_command(absolute_path, mutation_hash)
        output = run_command(str(then_run.get("command") or ""), then_run.get("timeout"))
        return json.dumps(
            {
                "mutation": mutation_result,
                "then_run": THEN_RUN_SUCCEEDED,
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
