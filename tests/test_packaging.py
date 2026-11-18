from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types as stdlib_types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PackagingTests(unittest.TestCase):
    def test_plugin_manifest_stays_at_root(self) -> None:
        text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
        self.assertIn("name: sol-pi", text)
        self.assertIn("obs_recall", text)

    def test_hermes_directory_load_exposes_register(self) -> None:
        ns_name = "_sol_pi_hermes_ns_test"
        module_name = f"{ns_name}.plugin"
        ns = stdlib_types.ModuleType(ns_name)
        ns.__path__ = []  # type: ignore[attr-defined]
        sys.modules[ns_name] = ns

        def _cleanup() -> None:
            for key in [n for n in sys.modules if n == ns_name or n.startswith(ns_name + ".")]:
                sys.modules.pop(key, None)

        self.addCleanup(_cleanup)
        spec = importlib.util.spec_from_file_location(
            module_name,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        self.assertIsNone(None) if spec is None else None
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.register))

    def _declared_entry_point(self) -> str:
        try:
            import tomllib
        except ModuleNotFoundError:
            self.skipTest("tomllib needs Python 3.11+")
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        return project["entry-points"]["hermes_agent.plugins"]["sol-pi"]

    def test_entry_point_declared_and_resolves_to_register(self) -> None:
        # Mirrors the host (plugins_loader): ep.load() yields either a module
        # (use its .register) or, for "module:function", the function itself.
        from sol_pi_hermes.plugin import register

        loaded = importlib.metadata.EntryPoint(
            "sol-pi", self._declared_entry_point(), "hermes_agent.plugins").load()
        self.assertIs(loaded if callable(loaded) else getattr(loaded, "register", None), register)

    def test_entry_point_package_is_listed_in_setuptools_packages(self) -> None:
        # Static check: setuptools must list the entry point target, or a built wheel would not ship it.
        top = self._declared_entry_point().split(":")[0].split(".")[0]
        import tomllib  # _declared_entry_point skips first on Python < 3.11

        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn(top, cfg["tool"]["setuptools"]["packages"])

    def _run_in_host(self, code: str, enable: bool = False, sol_pi_config: str | None = None) -> str:
        # Runs `code` under Hermes's own interpreter with a fake dist-info on a temp
        # path and a temp HERMES_HOME, so no real Hermes environment is modified.
        target = self._declared_entry_point()
        host = importlib.util.find_spec("hermes_cli")
        if host is None:
            if os.environ.get("SOL_PI_REQUIRE_HERMES") == "1":
                self.fail("hermes_cli is required by SOL_PI_REQUIRE_HERMES=1")
            self.skipTest("hermes_cli not importable")
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as site:
            dist = Path(site) / "sol_pi_hermes-0.0.dist-info"
            dist.mkdir()
            (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: sol-pi-hermes\nVersion: 0.0\n")
            (dist / "entry_points.txt").write_text(f"[hermes_agent.plugins]\nsol-pi = {target}\n")
            if sol_pi_config:
                (Path(home) / "sol-pi.json").write_text(sol_pi_config)
            if enable:
                (Path(home) / "config.yaml").write_text("plugins:\n  enabled: [sol-pi]\n")
            run = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=120,
                cwd=Path(host.origin).parents[1],  # host root holds sibling modules
                env={**os.environ, "HERMES_HOME": home,
                     "PYTHONPATH": os.pathsep.join([site, str(ROOT)])},
            )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return run.stdout

    def test_installed_entry_point_is_discovered_by_hermes(self) -> None:
        out = self._run_in_host(
            "from hermes_cli.plugins_discovery import discover_entrypoint_manifests as d;"
            "print([(m.name, m.source, m.path) for m in d()])")
        self.assertIn(f"('sol-pi', 'entrypoint', '{self._declared_entry_point()}')", out)

    def test_entry_point_plugin_loads_and_registers_in_hermes(self) -> None:
        # Discovery alone is not enough: with no plugin.yaml on this path, the
        # host must still load the plugin and accept its middleware and tool.
        out = self._run_in_host(
            "from hermes_cli.plugins import get_plugin_manager as g;"
            "m = g(); m.discover_and_load(); p = m._plugins.get('sol-pi');"
            "print('LOADED', bool(p and p.enabled), p and p.error);"
            "print('MW', len(m._middleware.get('llm_request', [])));"
            "print('TOOLS', sorted(getattr(p, 'tools_registered', []) or []))",
            enable=True)
        self.assertIn("LOADED True None", out)
        self.assertIn("MW 1", out)
        self.assertIn("TOOLS ['obs_recall']", out)

    def test_entry_point_plugin_registers_fusion_tools_when_enabled(self) -> None:
        out = self._run_in_host(
            "from hermes_cli.plugins import get_plugin_manager as g;"
            "m = g(); m.discover_and_load(); p = m._plugins.get('sol-pi');"
            "print('LOADED', bool(p and p.enabled), p and p.error);"
            "print('TOOLS', sorted(getattr(p, 'tools_registered', []) or []))",
            enable=True, sol_pi_config='{"actionFusion": true}')
        self.assertIn("LOADED True None", out)
        self.assertIn("TOOLS ['obs_recall', 'sol_pi_patch', 'sol_pi_write']", out)

    def test_no_typescript_imports(self) -> None:
        for path in ROOT.rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    self.assertNotIn(".ts", stripped)
                    self.assertNotIn("sol-pi/index", stripped)

    def test_real_hermes_plugins_validate_passes(self) -> None:
        if not shutil.which("hermes"):
            if os.environ.get("SOL_PI_REQUIRE_HERMES") == "1":
                self.fail("hermes CLI is required by SOL_PI_REQUIRE_HERMES=1")
            self.skipTest("hermes CLI not installed")
        with tempfile.TemporaryDirectory() as home:
            run = subprocess.run(
                ["hermes", "plugins", "validate", str(ROOT)],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "HERMES_HOME": home},
            )
        out = run.stdout + run.stderr
        self.assertEqual(run.returncode, 0, out)
        self.assertIn("Validation passed", out)
        self.assertNotIn("\u2717", out)
        # Pin that validate really checks the declared llm_request middleware
        # and hooks, so a Hermes that stops checking them cannot pass silently.
        for check in ("declared middleware", "declared hooks", "capability probe"):
            self.assertRegex(out, "\u2713 " + check)
        # Action Fusion is off by default, so the default probe legitimately
        # omits its tools; any *other* warning is a real manifest mismatch.
        for line in out.splitlines():
            if line.startswith("\u26a0"):
                self.assertIn("declares sol_pi_patch, sol_pi_write but", line)


if __name__ == "__main__":
    unittest.main()
