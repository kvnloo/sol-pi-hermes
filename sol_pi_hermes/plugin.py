"""Hermes ``register(ctx)`` for the SoL-Pi port."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .action_fusion import PATCH_SCHEMA, WRITE_SCHEMA, execute_mutation_then_run
from .config import SolPiConfig, load_sol_pi_config
from .observation_pack import (
    project_request,
    recall,
    runtime_root_for_session,
)

logger = logging.getLogger(__name__)

OBS_RECALL_SCHEMA = {
    "name": "obs_recall",
    "description": (
        "Read a stored large tool result by observation id and byte offset. "
        "Use the id from an ObservationPack placeholder; continue with next_offset."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Observation id from a placeholder"},
            "offset": {"type": "integer", "minimum": 0, "description": "Byte offset, default 0"},
        },
        "required": ["id"],
    },
}


def _session_id(**kwargs: Any) -> str:
    return str(kwargs.get("session_id") or kwargs.get("task_id") or "default")


def _cwd(**kwargs: Any) -> Path:
    cwd = kwargs.get("cwd") or kwargs.get("working_directory")
    return Path(cwd) if cwd else Path.cwd()


def register(ctx: Any, config: SolPiConfig | None = None) -> None:
    """Wire ObservationPack (llm_request middleware + obs_recall) and Action Fusion tools.

    Never registers a ContextEngine. EPR and OCC are ported in ``epr.py`` / ``occ.py`` but not
    wired: EPR needs a reducer model and OCC needs an ``agent_settled`` hook Hermes doesn't have.
    """
    cfg = config or load_sol_pi_config()
    sent_counts: dict[str, int] = {}
    roots: dict[str, Path] = {}

    def root_for(**kwargs: Any) -> Path:
        sid = _session_id(**kwargs)
        if sid not in roots:
            roots[sid] = runtime_root_for_session(sid)
        return roots[sid]

    def on_llm_request(request=None, **kwargs: Any):
        if not cfg.observation_pack or not isinstance(request, dict):
            return None
        return project_request(request, root_for(**kwargs), sent_counts)

    def obs_recall(args: dict, **kwargs: Any) -> str:
        return recall(root_for(**kwargs), str(args.get("id") or ""), int(args.get("offset") or 0))

    def _run_terminal(command: str, timeout: float | None) -> str:
        payload: dict[str, Any] = {"command": command}
        if timeout is not None:
            payload["timeout"] = timeout
        return str(ctx.dispatch_tool("terminal", payload))

    def sol_pi_write(args: dict, **kwargs: Any) -> str:
        path = Path(str(args.get("path") or ""))
        content = str(args.get("content") or "")

        def mutate() -> str:
            return str(ctx.dispatch_tool("write_file", {"path": str(path), "content": content}))

        if not cfg.action_fusion:
            return mutate()
        return execute_mutation_then_run(
            mutate=mutate,
            absolute_path=path if path.is_absolute() else _cwd(**kwargs) / path,
            then_run=args.get("then_run") if isinstance(args.get("then_run"), dict) else None,
            run_command=_run_terminal,
        )

    def sol_pi_patch(args: dict, **kwargs: Any) -> str:
        path = Path(str(args.get("path") or ""))

        def mutate() -> str:
            payload = {k: v for k, v in args.items() if k != "then_run"}
            return str(ctx.dispatch_tool("patch", payload))

        if not cfg.action_fusion:
            return mutate()
        return execute_mutation_then_run(
            mutate=mutate,
            absolute_path=path if path.is_absolute() else _cwd(**kwargs) / path,
            then_run=args.get("then_run") if isinstance(args.get("then_run"), dict) else None,
            run_command=_run_terminal,
        )

    ctx.register_tool(name="obs_recall", toolset="sol-pi", schema=OBS_RECALL_SCHEMA, handler=obs_recall)
    ctx.register_tool(name="sol_pi_write", toolset="sol-pi", schema=WRITE_SCHEMA, handler=sol_pi_write)
    ctx.register_tool(name="sol_pi_patch", toolset="sol-pi", schema=PATCH_SCHEMA, handler=sol_pi_patch)
    ctx.register_middleware("llm_request", on_llm_request)
