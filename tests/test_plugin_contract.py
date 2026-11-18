"""Keep the plugin's directory manifest, wheel entry point and runtime wiring aligned."""

from __future__ import annotations

import importlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.plugin import register as implementation_register

ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_FIELDS = ("provides_tools", "provides_hooks", "provides_middleware")


def manifest_contract() -> dict[str, object]:
    """Read the scalar and block-list fields used by this dependency-free fixture."""
    fields: dict[str, object] = {}
    current_list: str | None = None
    for raw in (ROOT / "plugin.yaml").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not raw[0].isspace():
            key, colon, value = raw.partition(":")
            if not colon:
                continue
            current_list = None
            if key in CAPABILITY_FIELDS:
                if value.strip():
                    raise AssertionError(f"{key} must use a YAML block list")
                fields[key] = []
                current_list = key
            elif key in ("name", "version"):
                fields[key] = value.strip().strip("\"'")
        elif current_list is not None:
            if not line.startswith("- "):
                raise AssertionError(f"Unexpected item in {current_list}: {line}")
            fields[current_list].append(line[2:].split(" #", 1)[0].strip())
    for required in ("name", "version", "provides_tools", "provides_middleware"):
        if required not in fields:
            raise AssertionError(f"Missing manifest field {required}")
    fields.setdefault("provides_hooks", [])
    return fields


class RecordingContext:
    def __init__(self) -> None:
        self.tools: dict[str, tuple[dict, object, str]] = {}
        self.hooks: list[str] = []
        self.middleware: dict[str, object] = {}
        self.context_engines: list[object] = []

    def register_tool(self, *, name, schema, handler, toolset) -> None:
        if name in self.tools:
            raise AssertionError(f"Duplicate tool: {name}")
        self.tools[name] = (schema, handler, toolset)

    def register_hook(self, name, *args, **kwargs) -> None:
        self.hooks.append(name)

    def register_middleware(self, kind, callback) -> None:
        if kind in self.middleware:
            raise AssertionError(f"Duplicate middleware: {kind}")
        self.middleware[kind] = callback

    def register_context_engine(self, engine) -> None:
        self.context_engines.append(engine)


class PluginContractTests(unittest.TestCase):
    def test_manifest_entry_point_and_registered_capabilities(self) -> None:
        manifest = manifest_contract()
        # Regex, not tomllib: requires-python is >=3.10 and tomllib is 3.11+.
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"', pyproject, re.M)
        self.assertEqual(version and version[1], manifest["version"])

        # Hermes main discovers installed plugins in the hermes_agent.plugins group.
        section = re.search(r'^\[project\.entry-points\."hermes_agent\.plugins"\]\n((?:[^\[\n].*\n?)*)',
                            pyproject, re.M)
        self.assertIsNotNone(section)
        entry_points = dict(re.findall(r'^([\w.-]+) = "([^"]+)"', section[1], re.M))
        self.assertEqual(set(entry_points), {manifest["name"]})
        module_name, separator, attr = entry_points[manifest["name"]].partition(":")
        self.assertEqual(separator, ":")
        self.assertTrue(module_name)
        self.assertTrue(attr)
        entry_register = getattr(importlib.import_module(module_name), attr)
        self.assertIs(entry_register, implementation_register)

        advertised = set(manifest["provides_tools"])
        self.assertEqual(advertised, {"obs_recall", "sol_pi_write", "sol_pi_patch"})
        self.assertEqual(len(advertised), len(manifest["provides_tools"]))
        self.assertEqual(set(manifest["provides_middleware"]), {"llm_request"})
        for config, expected_tools in (
            (SolPiConfig(), {"obs_recall"}),
            (SolPiConfig(observation_pack=True), {"obs_recall"}),
            (SolPiConfig(action_fusion=True), advertised),
            (SolPiConfig(action_fusion=True, observation_pack=True), advertised),
        ):
            with self.subTest(action_fusion=config.action_fusion, observation_pack=config.observation_pack):
                ctx = RecordingContext()
                entry_register(ctx, config)
                self.assertEqual(set(ctx.tools), expected_tools)
                self.assertEqual(set(ctx.hooks), set(manifest["provides_hooks"]))
                self.assertEqual(set(ctx.middleware), set(manifest["provides_middleware"]))
                self.assertEqual(ctx.context_engines, [])
                for name, (schema, handler, toolset) in ctx.tools.items():
                    self.assertEqual(schema["name"], name)
                    self.assertEqual(toolset, manifest["name"])
                    self.assertTrue(callable(handler))
                for callback in ctx.middleware.values():
                    self.assertTrue(callable(callback))

    def test_host_style_register_without_config_reads_sol_pi_json(self) -> None:
        # register() called with no config argument: flags must come from the real config file.
        for raw, expected in (
            (None, {"obs_recall"}),
            ({"version": 1, "actionFusion": True}, {"obs_recall", "sol_pi_write", "sol_pi_patch"}),
        ):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as home:
                if raw is not None:
                    (Path(home) / "sol-pi.json").write_text(json.dumps(raw), encoding="utf-8")
                ctx = RecordingContext()
                with patch("sol_pi_hermes.config._hermes_home", return_value=Path(home)):
                    implementation_register(ctx)
                self.assertEqual(set(ctx.tools), expected)

    def test_observation_pack_flag_gates_projection_not_middleware_registration(self) -> None:
        disabled = RecordingContext()
        enabled = RecordingContext()
        implementation_register(disabled, SolPiConfig())
        implementation_register(enabled, SolPiConfig(observation_pack=True))
        request = {"messages": [{"role": "user", "content": "hello"}]}
        projected = {"request": {"messages": []}}
        with (
            patch("sol_pi_hermes.plugin.runtime_root_for_session", return_value=ROOT),
            patch("sol_pi_hermes.plugin.project_request", return_value=projected) as project,
        ):
            self.assertIsNone(disabled.middleware["llm_request"](request=request, session_id="case"))
            project.assert_not_called()
            self.assertIs(
                enabled.middleware["llm_request"](request=request, session_id="case"),
                projected,
            )
            project.assert_called_once()


if __name__ == "__main__":
    unittest.main()
