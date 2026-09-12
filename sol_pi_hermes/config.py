"""SoL-Pi feature flags for the Hermes host.

Defaults match NVlabs/SoL-Pi: every mechanism is off until config enables it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

FEATURE_KEYS = (
    "actionFusion",
    "observationPack",
    "evidencePreservingReducer",
    "onlineContextCompact",
)
STRING_KEYS = (
    "evidencePreservingReducerModel",
    "evidencePreservingReducerProvider",
)
CONFIG_KEYS = {"version", *FEATURE_KEYS, *STRING_KEYS, "cacheWriteReadRatio"}
DEFAULT_CACHE_WRITE_READ_RATIO = 12.5
DEFAULT_REDUCER_MODEL = "gpt-5.6-luna"
DEFAULT_REDUCER_PROVIDER = "openai-codex"


@dataclass(frozen=True)
class SolPiConfig:
    version: int = 1
    action_fusion: bool = False
    observation_pack: bool = False
    evidence_preserving_reducer: bool = False
    evidence_preserving_reducer_model: str = DEFAULT_REDUCER_MODEL
    evidence_preserving_reducer_provider: str = DEFAULT_REDUCER_PROVIDER
    online_context_compact: bool = False
    cache_write_read_ratio: float = DEFAULT_CACHE_WRITE_READ_RATIO


def _hermes_home() -> Path:
    return Path.home() / ".hermes"


def find_config_path(cwd: str | Path | None = None, hermes_home: Path | None = None) -> Path | None:
    home = hermes_home or _hermes_home()
    if cwd is not None:
        project = Path(cwd) / ".hermes" / "sol-pi.json"
        if project.is_file():
            return project
    global_path = home / "sol-pi.json"
    return global_path if global_path.is_file() else None


def load_sol_pi_config(
    cwd: str | Path | None = None,
    hermes_home: Path | None = None,
    *,
    raw: Mapping[str, Any] | None = None,
) -> SolPiConfig:
    if raw is None:
        path = find_config_path(cwd, hermes_home)
        if path is None:
            return SolPiConfig()
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read SoL-Pi config {path}: {exc}") from exc
    else:
        parsed = raw

    if not isinstance(parsed, dict):
        raise ValueError("SoL-Pi config must be a JSON object")
    for key in parsed:
        if key not in CONFIG_KEYS:
            raise ValueError(f"Unknown SoL-Pi config key: {key}")
    version = parsed.get("version", 1)
    if version != 1:
        raise ValueError("SoL-Pi config version must be 1")

    def _bool(key: str) -> bool:
        value = parsed.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"SoL-Pi config {key} must be boolean")
        return value

    ratio = parsed.get("cacheWriteReadRatio", DEFAULT_CACHE_WRITE_READ_RATIO)
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio < 0:
        raise ValueError("SoL-Pi config cacheWriteReadRatio must be a finite non-negative number")

    def _str(key: str, default: str) -> str:
        value = parsed.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SoL-Pi config {key} must be a non-empty string")
        return value

    return SolPiConfig(
        version=1,
        action_fusion=_bool("actionFusion"),
        observation_pack=_bool("observationPack"),
        evidence_preserving_reducer=_bool("evidencePreservingReducer"),
        evidence_preserving_reducer_model=_str(
            "evidencePreservingReducerModel", DEFAULT_REDUCER_MODEL
        ),
        evidence_preserving_reducer_provider=_str(
            "evidencePreservingReducerProvider", DEFAULT_REDUCER_PROVIDER
        ),
        online_context_compact=_bool("onlineContextCompact"),
        cache_write_read_ratio=float(ratio),
    )
