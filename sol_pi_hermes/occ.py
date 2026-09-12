"""Online Context Compact: no-op unless the host emits agent_settled.

SoL-Pi only starts boundary compaction after Pi's ``agent_settled``. Hermes
``post_llm_call`` is not a substitute (retries / continuations can still run).
This module records plan-boundary intent and refuses compact() without settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OnlineCompactGate:
    enabled: bool = False
    settled: bool = False
    pending_compact: bool = False
    last_reason: str = "waiting_for_agent_settled"

    def on_plan_boundary(self) -> None:
        if not self.enabled:
            self.last_reason = "disabled"
            return
        self.pending_compact = True
        if not self.settled:
            self.last_reason = "waiting_for_agent_settled"

    def on_agent_settled(self) -> bool:
        """Return True when compaction around the built-in compressor may run."""
        self.settled = True
        if not self.enabled:
            self.last_reason = "disabled"
            return False
        if not self.pending_compact:
            self.last_reason = "no_pending_boundary"
            return False
        self.last_reason = "compact_via_builtin_compressor"
        self.pending_compact = False
        return True

    def should_compact(self) -> bool:
        return self.enabled and self.settled and self.pending_compact
