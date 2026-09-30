"""Hermes ``register(ctx)`` for the SoL-Pi port."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .action_fusion import PATCH_SCHEMA, WRITE_SCHEMA, execute_mutation_then_run
from .config import SolPiConfig, load_sol_pi_config
from .epr import transform_if_reduced
from .observation_pack import (
    project_messages,
    recall,
    runtime_root_for_session,
)
from .occ import OnlineCompactGate

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
    return str(kwargs.get("session_id") or kwargs.get("task_id") or "")


def _cwd(**kwargs: Any) -> Path:
    cwd = kwargs.get("cwd") or kwargs.get("working_directory")
    return Path(cwd) if cwd else Path.cwd()


def register(ctx: Any, config: SolPiConfig | None = None) -> None:
    """Wire tools and hooks. Never registers a ContextEngine."""
    cfg = config or load_sol_pi_config()
    sent_counts: dict[str, int] = {}
    gate = OnlineCompactGate(enabled=cfg.online_context_compact)

    def root_for(**kwargs: Any) -> Path:
        # Both request projection and tool dispatch run in the owning profile.
        # Resolve on every call: one process can serve several homes/sessions.
        return runtime_root_for_session(_session_id(**kwargs))

    def on_llm_request(request=None, **kwargs: Any):
        if not isinstance(request, dict) or not _session_id(**kwargs):
            return
        messages = request.get("messages")
        if "input" in request or not isinstance(messages, list) or not messages:
            return
        if not all(isinstance(message, dict) for message in messages):
            return
        # This seam runs after provider decoration. Do not flatten text blocks
        # (which can carry cache metadata), images, or native provider envelopes.
        if any(message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
               or (message.get("role") == "tool" and not isinstance(message.get("content"), str))
               for message in messages):
            return
        projected = project_messages(messages, root_for(**kwargs), sent_counts)
        if projected == messages:
            return
        return {"request": {**request, "messages": projected}, "source": "sol-pi",
                "reason": "ObservationPack request-only projection"}

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

    def on_transform_tool_result(tool_name=None, args=None, result=None, **kwargs: Any):
        if not cfg.evidence_preserving_reducer:
            return None
        args = args if isinstance(args, dict) else {}
        text = result if isinstance(result, str) else json.dumps(result)
        return transform_if_reduced(tool_name=str(tool_name or ""), args=args, result=text, reducer=None)

    def on_agent_settled(**kwargs: Any) -> None:
        allowed = gate.on_agent_settled()
        if allowed:
            logger.info("sol-pi OCC: agent_settled; compact via built-in compressor only")
        else:
            logger.debug("sol-pi OCC skipped: %s", gate.last_reason)

    ctx.register_tool(name="obs_recall", toolset="sol-pi", schema=OBS_RECALL_SCHEMA, handler=obs_recall)
    ctx.register_tool(name="sol_pi_write", toolset="sol-pi", schema=WRITE_SCHEMA, handler=sol_pi_write)
    ctx.register_tool(name="sol_pi_patch", toolset="sol-pi", schema=PATCH_SCHEMA, handler=sol_pi_patch)
    if cfg.observation_pack:
        ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_hook("transform_tool_result", on_transform_tool_result)
    ctx.register_hook("agent_settled", on_agent_settled)
