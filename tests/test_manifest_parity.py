import re
import unittest
from pathlib import Path

from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.plugin import register

MANIFEST = (Path(__file__).resolve().parents[1] / "plugin.yaml").read_text(encoding="utf-8")


def declared(key):
    block = re.search(rf"^{key}:\n((?:  - .+\n?)+)", MANIFEST, re.M)
    return {line[4:].strip() for line in block.group(1).splitlines()}


class Ctx:
    def __init__(self):
        self.tools, self.middleware = set(), set()

    def register_tool(self, **kw):
        self.tools.add(kw["name"])

    def register_middleware(self, name, handler):
        self.middleware.add(name)


class ManifestParityTests(unittest.TestCase):
    def test_full_config_registers_exactly_what_plugin_yaml_declares(self):
        ctx = Ctx()
        register(ctx, SolPiConfig(action_fusion=True))
        self.assertEqual(ctx.tools, declared("provides_tools"))
        self.assertEqual(ctx.middleware, declared("provides_middleware"))


if __name__ == "__main__":
    unittest.main()
