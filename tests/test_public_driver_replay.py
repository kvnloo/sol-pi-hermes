import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/replay_public_driver.py'
spec = importlib.util.spec_from_file_location('public_replay', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class PublicReplayTests(unittest.TestCase):
    def test_digest_pin(self):
        m.verify_digest(b'sample', hashlib.sha256(b'sample').hexdigest())
        with self.assertRaises(ValueError):
            m.verify_digest(b'other', hashlib.sha256(b'sample').hexdigest())

    def test_wrong_pin_rejected_under_optimization(self):
        code = f"import runpy; m=runpy.run_path({str(SCRIPT)!r}); m['verify_digest'](b'changed','wrong')"
        p = subprocess.run([sys.executable, '-O', '-B', '-c', code], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('ValueError: Donor bytes differ from pin', p.stderr)

    def test_host_pin_and_cleanliness(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            for replies, valid in [([str(root),m.HOST,''], True),
                                   ([str(root),'wrong',''], False),
                                   ([str(root),m.HOST,' M tracked.py'], False)]:
                with patch.object(m.subprocess, 'check_output', side_effect=replies):
                    if valid:
                        self.assertEqual(m.verify_host(root),root)
                    else:
                        with self.assertRaises(ValueError): m.verify_host(root)

    def test_import_origin(self):
        module=types.SimpleNamespace(__file__='/tmp/expected/module.py',__name__='example')
        m.verify_import(module,Path('/tmp/expected'),'module.py')
        with self.assertRaises(ValueError): m.verify_import(module,Path('/tmp/other'),'module.py')

    def test_spill_boundary(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as other:
            good=Path(home)/'spill.json'; good.write_bytes(b'public')
            outside=Path(other)/'spill.json'; outside.write_bytes(b'other')
            self.assertEqual(m.spill_bytes(good,home),b'public')
            with self.assertRaises(ValueError): m.spill_bytes(outside,home)
            with self.assertRaises(ValueError): m.spill_bytes(home,home)
