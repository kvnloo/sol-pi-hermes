import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from sol_pi_hermes.config import _hermes_home
from sol_pi_hermes.plugin import register


class ProfileFailureTests(unittest.TestCase):
    def test_profile_resolution_failure_cannot_load_default_enabled_policy(self):
        for failure in (RuntimeError('profile unavailable'), OSError('profile IO failed'),
                        ImportError('host dependency failed')):
            with self.subTest(error=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                (home / '.hermes').mkdir()
                (home / '.hermes' / 'sol-pi.json').write_text(json.dumps({'actionFusion': True}))
                host = types.ModuleType('hermes_constants')
                host.get_hermes_home = Mock(side_effect=failure)
                ctx = Mock()
                with patch.dict(sys.modules, {'hermes_constants': host}), patch.object(Path, 'home', return_value=home):
                    with self.assertRaises(type(failure)):
                        register(ctx)
                ctx.register_tool.assert_not_called()
                ctx.register_middleware.assert_not_called()

    def test_present_host_without_resolver_is_not_standalone(self):
        host = types.ModuleType('hermes_constants')
        with patch.dict(sys.modules, {'hermes_constants': host}), patch.object(Path, 'home', side_effect=AssertionError('fallback')):
            with self.assertRaises(ImportError):
                _hermes_home()

    def test_missing_transitive_module_is_not_standalone(self):
        import builtins
        real_import = builtins.__import__
        def importing(name, *args, **kwargs):
            if name == 'hermes_constants':
                raise ModuleNotFoundError("No module named 'host_dependency'", name='host_dependency')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=importing), patch.object(Path, 'home', side_effect=AssertionError('fallback')):
            with self.assertRaises(ModuleNotFoundError):
                _hermes_home()
