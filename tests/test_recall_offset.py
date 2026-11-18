import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.plugin import register

class Context:
    def register_tool(self, **kw): self.recall = kw['handler']
    def register_middleware(self, *args): pass

class RecallOffsetTests(unittest.TestCase):
    def test_invalid_offsets_fail_before_archive_access(self):
        ctx = Context(); register(ctx, SolPiConfig(observation_pack=True))
        for offset in (True, False, 1.9, 2.0, -1, '2', 'garbage', None, [], {}, float('inf')):
            with self.subTest(offset=offset), patch('sol_pi_hermes.plugin.runtime_root_for_session') as root, patch('sol_pi_hermes.plugin.recall', return_value='{}') as read:
                result = json.loads(ctx.recall({'id':'synthetic', 'offset':offset}, session_id='s'))
                self.assertIn('error', result)
                root.assert_not_called(); read.assert_not_called()

    def test_default_and_nonnegative_integer_offsets_preserved(self):
        ctx = Context(); register(ctx, SolPiConfig(observation_pack=True))
        for args, expected in (({'id':'synthetic'},0), ({'id':'synthetic','offset':0},0), ({'id':'synthetic','offset':17},17)):
            with self.subTest(args=args), patch('sol_pi_hermes.plugin.runtime_root_for_session',return_value=Path('/unused')), patch('sol_pi_hermes.plugin.recall',return_value='{}') as read:
                self.assertEqual(ctx.recall(args,session_id='s'),'{}')
                read.assert_called_once_with(Path('/unused'),'synthetic',expected)

    def test_huge_integer_offset_is_json_error_not_exception(self):
        from sol_pi_hermes.observation_pack import OBSERVATION_ID_PATTERN, observation_path
        oid = 'obs_' + '0' * 24
        self.assertTrue(OBSERVATION_ID_PATTERN.fullmatch(oid))
        ctx = Context(); register(ctx, SolPiConfig(observation_pack=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = observation_path(Path(tmp), oid); path.parent.mkdir(parents=True); path.write_text('x')
            with patch('sol_pi_hermes.plugin.runtime_root_for_session', return_value=Path(tmp)):
                result = json.loads(ctx.recall({'id': oid, 'offset': 10**30}, session_id='s'))
        self.assertIn('exceeds observation size', result['error'])
