from __future__ import annotations

import importlib.util
import os
import re
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

    def test_no_typescript_imports(self) -> None:
        for path in ROOT.rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    self.assertNotIn(".ts", stripped)
                    self.assertNotIn("sol-pi/index", stripped)

    def test_entry_point_plugin_loads_and_registers_in_hermes(self) -> None:
        # The declared module target must load in the real host (no plugin.yaml
        # on this path) and register its middleware and tool.
        host = importlib.util.find_spec("hermes_cli")
        if host is None:
            if os.environ.get("SOL_PI_REQUIRE_HERMES") == "1":
                self.fail("hermes_cli is required by SOL_PI_REQUIRE_HERMES=1")
            self.skipTest("hermes_cli not importable")
        target = re.search(r'^sol-pi = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)[1]
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as site:
            dist = Path(site, "sol_pi_hermes-0.0.dist-info")
            dist.mkdir()
            dist.joinpath("METADATA").write_text("Metadata-Version: 2.1\nName: sol-pi-hermes\nVersion: 0.0\n")
            dist.joinpath("entry_points.txt").write_text(f"[hermes_agent.plugins]\nsol-pi = {target}\n")
            Path(home, "config.yaml").write_text("plugins:\n  enabled: [sol-pi]\n")
            run = subprocess.run(
                [sys.executable, "-c",
                 "from hermes_cli.plugins import get_plugin_manager as g;"
                 "m = g(); m.discover_and_load(); p = m._plugins.get('sol-pi');"
                 "print('LOADED', bool(p and p.enabled), p and p.error);"
                 "print('MW', len(m._middleware.get('llm_request', [])));"
                 "print('TOOLS', sorted(p.tools_registered or []))"],
                capture_output=True, text=True, timeout=120, cwd=Path(host.origin).parents[1],
                env={**os.environ, "HERMES_HOME": home, "PYTHONPATH": os.pathsep.join([site, str(ROOT)])})
        out = run.stdout + run.stderr
        self.assertEqual(run.returncode, 0, out)
        self.assertIn("LOADED True None", out)
        self.assertIn("MW 1", out)
        self.assertIn("TOOLS ['obs_recall']", out)

    def test_installed_wheel_exposes_hermes_plugin_entry_point(self) -> None:
        if not all(importlib.util.find_spec(name) for name in ("pip", "setuptools", "wheel")):
            self.skipTest("offline wheel build needs pip, setuptools and wheel")
        with tempfile.TemporaryDirectory() as tmp:
            src, target = Path(tmp, "src"), Path(tmp, "site")
            shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(".git", "build", "*.egg-info", "__pycache__"))
            install = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                 "--no-index", "--target", str(target), str(src)],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(install.returncode, 0, install.stdout + install.stderr)
            found = subprocess.run(
                [sys.executable, "-c",
                 "import sys; sys.path.insert(0, sys.argv[1]);"
                 "from importlib.metadata import distributions;"
                 "eps = [e for d in distributions(path=[sys.argv[1]]) for e in d.entry_points"
                 " if e.group == 'hermes_agent.plugins'];"
                 "assert [(e.name, e.value) for e in eps] == [('sol-pi', 'sol_pi_hermes')], eps;"
                 "import sol_pi_hermes; assert sol_pi_hermes.__file__.startswith(sys.argv[1]);"
                 "loaded = eps[0].load();"
                 "from sol_pi_hermes.plugin import register;"
                 "assert loaded.register is register and callable(loaded.register); print('ok')", str(target)],
                capture_output=True, text=True, timeout=60, cwd=tmp)
        self.assertEqual(found.returncode, 0, found.stdout + found.stderr)
        self.assertEqual(found.stdout.strip(), "ok", found.stderr)


class EvaluatorPackagingTests(unittest.TestCase):
    def test_default_fixtures_belong_to_runtime_package(self):
        from sol_pi_hermes import observation_eval
        self.assertEqual(observation_eval.DEFAULT_FIXTURES.parent,
                         Path(observation_eval.__file__).resolve().parent / 'fixtures')
        self.assertTrue(observation_eval.DEFAULT_FIXTURES.is_file())

    def test_wheel_package_data_ships_the_fixture(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("tomllib needs Python 3.11+")
        root = Path(__file__).resolve().parents[1]
        cfg = tomllib.loads((root / "pyproject.toml").read_text())
        globs = cfg["tool"]["setuptools"]["package-data"]["sol_pi_hermes"]
        from sol_pi_hermes import observation_eval
        pkg = observation_eval.DEFAULT_FIXTURES.parents[1]
        shipped = {p for g in globs for p in pkg.glob(g)}
        self.assertIn(observation_eval.DEFAULT_FIXTURES, shipped)


if __name__ == "__main__":
    unittest.main()
