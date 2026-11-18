import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.config import find_config_path
from sol_pi_hermes.plugin import register


class Ctx:
    def __init__(self):
        self.tools = {}

    def register_tool(self, **kw):
        self.tools[kw['name']] = kw['handler']

    def register_middleware(self, *args):
        pass


def write(root, flag, where='.hermes'):
    d = Path(root) / where
    d.mkdir(parents=True, exist_ok=True)
    (d / 'sol-pi.json').write_text(json.dumps({'version': 1, 'actionFusion': flag}))
    return d / 'sol-pi.json'


class ConfigPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__('shutil').rmtree, self.tmp, True)
        self.home, self.proj = self.tmp / 'home', self.tmp / 'proj'
        self.home.mkdir(), self.proj.mkdir()

    def test_project_beats_global_and_global_is_fallback(self):
        g = write(self.home, True, '.')
        self.assertEqual(find_config_path(self.proj, self.home), g)
        p = write(self.proj, False)
        self.assertEqual(find_config_path(self.proj, self.home), p)
        self.assertIsNone(find_config_path(self.tmp / 'bare', self.tmp / 'empty'))

    def _register_in_project(self):
        ctx = Ctx()
        old = os.getcwd()
        self.addCleanup(os.chdir, old)
        os.chdir(self.proj)
        with patch('sol_pi_hermes.config._hermes_home', return_value=self.home):
            register(ctx)
        return set(ctx.tools)

    def test_repo_project_file_cannot_enable_fusion_at_registration(self):
        write(self.proj, True)
        self.assertEqual(self._register_in_project(), {'obs_recall'})

    def test_hermes_home_file_enables_fusion_at_registration(self):
        write(self.home, True, '.')
        self.assertGreater(self._register_in_project(), {'obs_recall'})
