from __future__ import annotations

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

    def test_no_typescript_imports(self) -> None:
        for path in ROOT.rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    self.assertNotIn(".ts", stripped)
                    self.assertNotIn("sol-pi/index", stripped)

    @unittest.skipUnless(shutil.which("hermes"), "hermes CLI not installed")
    def test_real_hermes_plugins_validate_passes(self) -> None:
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
