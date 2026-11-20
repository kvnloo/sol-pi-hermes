"""Hermes ``register(ctx)`` for the SoL-Pi port."""

from __future__ import annotations

import json
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


def _session_id(**kwargs: Any) -> str | None:
    # Never map missing host context to a real session named "default".
    for key in ("session_id", "task_id"):
        value = kwargs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _tool_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Only host-provided routing identities are nested-dispatch capabilities.
    return {key: kwargs[key] for key in ("task_id", "session_id") if key in kwargs}


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
        if sid is None:
            raise ValueError("ObservationPack requires a host session_id or task_id")
        if sid not in roots:
            roots[sid] = runtime_root_for_session(sid)
        return roots[sid]

    def on_llm_request(request=None, **kwargs: Any):
        if not cfg.observation_pack or not isinstance(request, dict):
            return None
        if _session_id(**kwargs) is None:
            return None  # Preserve the full observation; do not persist it unscoped.
        return project_request(request, root_for(**kwargs), sent_counts)

    def obs_recall(args: dict, **kwargs: Any) -> str:
        if _session_id(**kwargs) is None:
            return json.dumps({"error": "ObservationPack requires a host session_id or task_id"})
        offset = args.get("offset", 0)
        if type(offset) is not int or offset < 0:
            return json.dumps({"error": "ObservationPack offset must be a non-negative integer"})
        return recall(root_for(**kwargs), str(args.get("id") or ""), offset)

    def _run_terminal(command: str, timeout: float | None, context: dict[str, Any]) -> str:
        payload: dict[str, Any] = {"command": command}
        if timeout is not None:
            payload["timeout"] = timeout
        return str(ctx.dispatch_tool("terminal", payload, **context))

    def sol_pi_write(args: dict, **kwargs: Any) -> str:
        # Plugin schemas are not runtime argument validation in Hermes. Never
        # coerce absent/invalid content into an intentional empty-file write.
        if not isinstance(args, dict):
            return json.dumps({"error": "sol_pi_write requires an argument object; no mutation was attempted."})
        if not isinstance(args.get("path"), str) or not args["path"]:
            return json.dumps({"error": "sol_pi_write requires a nonempty string path; no mutation was attempted."})
        if not isinstance(args.get("content"), str):
            return json.dumps({"error": "sol_pi_write requires string content (empty string is allowed); no mutation was attempted."})
        path = Path(args["path"])
        context = _tool_context(kwargs)
        content = args["content"]

        def mutate() -> str:
            return str(ctx.dispatch_tool("write_file", {"path": str(path), "content": content}, **context))

        if not cfg.action_fusion:
            return mutate()
        return execute_mutation_then_run(
            mutate=mutate,
            # PluginContext does not attest that the backend path is host-local.
            absolute_path=None,
            then_run=args.get("then_run"),
            run_command=lambda command, timeout: _run_terminal(command, timeout, context),
        )

    def sol_pi_patch(args: dict, **kwargs: Any) -> str:
        path = Path(str(args.get("path") or ""))
        context = _tool_context(kwargs)

        def mutate() -> str:
            payload = {k: v for k, v in args.items() if k != "then_run"}
            return str(ctx.dispatch_tool("patch", payload, **context))

        if not cfg.action_fusion:
            return mutate()
        return execute_mutation_then_run(
            mutate=mutate,
            # PluginContext does not attest that the backend path is host-local.
            absolute_path=None,
            then_run=args.get("then_run"),
            run_command=lambda command, timeout: _run_terminal(command, timeout, context),
        )

    ctx.register_tool(name="obs_recall", toolset="sol-pi", schema=OBS_RECALL_SCHEMA, handler=obs_recall)
    if cfg.action_fusion:
        ctx.register_tool(name="sol_pi_write", toolset="sol-pi", schema=WRITE_SCHEMA, handler=sol_pi_write)
        ctx.register_tool(name="sol_pi_patch", toolset="sol-pi", schema=PATCH_SCHEMA, handler=sol_pi_patch)
    ctx.register_middleware("llm_request", on_llm_request)
