from __future__ import annotations

import importlib.util
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
                 "assert [(e.name, e.value) for e in eps] == [('sol-pi', 'sol_pi_hermes:register')], eps;"
                 "import sol_pi_hermes; assert sol_pi_hermes.__file__.startswith(sys.argv[1]);"
                 "assert callable(eps[0].load()); print('ok')", str(target)],
                capture_output=True, text=True, timeout=60, cwd=tmp)
        self.assertEqual(found.stdout.strip(), "ok", found.stderr)


if __name__ == "__main__":
    unittest.main()
