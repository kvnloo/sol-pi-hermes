from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types as stdlib_types
import unittest
from unittest import mock
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

    def _run_validate_test_without_hermes(self, flag: str | None) -> unittest.TestResult:
        env = {k: v for k, v in os.environ.items() if k != "SOL_PI_REQUIRE_HERMES"}
        if flag is not None:
            env["SOL_PI_REQUIRE_HERMES"] = flag
        result = unittest.TestResult()
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("shutil.which", return_value=None):
            PackagingTests("test_real_hermes_plugins_validate_passes").run(result)
        return result

    def test_require_hermes_flag_fails_instead_of_skipping(self) -> None:
        result = self._run_validate_test_without_hermes("1")
        self.assertEqual(len(result.failures), 1, result.skipped)
        self.assertEqual(result.skipped, [])
        self.assertEqual(result.errors, [])

    def test_missing_hermes_skips_when_flag_unset(self) -> None:
        result = self._run_validate_test_without_hermes(None)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual((result.failures, result.errors), ([], []))


if __name__ == "__main__":
    unittest.main()
