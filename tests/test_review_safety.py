import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.action_fusion import execute_mutation_then_run
from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.observation_pack import append_ledger, ledger_path
from sol_pi_hermes.plugin import register


class Context:
    def __init__(self):
        self.tools = {}
        self.calls = []

    def register_tool(self, **kw):
        self.tools[kw['name']] = kw['handler']

    def register_middleware(self, *args):
        pass

    def dispatch_tool(self, name, args, **kwargs):
        self.calls.append((name, args, kwargs))
        return json.dumps({'exit_code': 0} if name == 'terminal' else {'success': True, 'path': '/remote/result'})


class ReviewSafetyTests(unittest.TestCase):
    def test_disabled_fusion_has_no_mutation_tools(self):
        ctx = Context()
        register(ctx, SolPiConfig())
        self.assertEqual(set(ctx.tools), {'obs_recall'})

    def test_scoped_dispatch_and_remote_path_never_read_host(self):
        ctx = Context()
        register(ctx, SolPiConfig(action_fusion=True))
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('host read')):
            for sid in ('one', 'two'):
                for tool in ('sol_pi_write', 'sol_pi_patch'):
                    result = json.loads(ctx.tools[tool](
                        {'path': 'relative.txt', 'content': 'x', 'then_run': {'command': 'test-only', 'timeout': 2}},
                        task_id='task-' + sid, session_id=sid, cwd='/wrong-host', unrelated='not-forwarded'))
                    self.assertEqual(result['then_run'], '[then_run:succeeded]')
                    for _, _, scope in ctx.calls[-2:]:
                        self.assertEqual(scope, {'task_id': 'task-' + sid, 'session_id': sid})
        self.assertEqual(len(ctx.calls), 8)

    def test_no_then_run_preserves_original_result(self):
        result = '{"error":"denied"}'
        self.assertEqual(execute_mutation_then_run(mutate=lambda: result, absolute_path=None,
            then_run=None, run_command=lambda *_: self.fail('unexpected dispatch')), result)

    def test_returned_mutation_failures_skip_command(self):
        for payload in ({'error': 'denied'}, {'success': False}, {'exit_code': 1}):
            with self.subTest(payload=payload):
                result = json.loads(execute_mutation_then_run(mutate=lambda: json.dumps(payload), absolute_path=None,
                    then_run={'command': 'never'}, run_command=lambda *_: self.fail('unexpected dispatch')))
                self.assertEqual(result['then_run'], '[then_run:skipped]')

    def test_returned_terminal_failures_are_not_success(self):
        for payload in ({'error': 'denied'}, {'success': False}, {'exit_code': 1}, {'exit_code': -1}, {'exit_code': -9}):
            with self.subTest(payload=payload):
                result = json.loads(execute_mutation_then_run(mutate=lambda: '{"success":true}', absolute_path=None,
                    then_run={'command': 'fixture'}, run_command=lambda *_: json.dumps(payload)))
                self.assertEqual(result['then_run'], '[then_run:failed]')

    def test_background_or_incomplete_result_is_unknown(self):
        for payload in ({'session_id': 'job', 'status': 'running'}, {'exit_code': None}, {}, None, [], 'plain output', {'exit_code': False}):
            with self.subTest(payload=payload):
                result = json.loads(execute_mutation_then_run(mutate=lambda: '{"success":true}', absolute_path=None,
                    then_run={'command': 'fixture'}, run_command=lambda *_: json.dumps(payload)))
                self.assertEqual(result['then_run'], '[then_run:unknown]')

    def test_ledger_mode_and_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = os.umask(0o022)
            try:
                append_ledger(root, {'event': 'one'})
                append_ledger(root, {'event': 'two'})
            finally:
                os.umask(old)
            path = ledger_path(root)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual([json.loads(x)['event'] for x in path.read_text().splitlines()], ['one', 'two'])

    def test_ledger_rejects_symlink_without_touching_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target'
            target.write_text('unchanged')
            path = ledger_path(root)
            path.parent.mkdir()
            path.symlink_to(target)
            with self.assertRaises(OSError):
                append_ledger(root, {'event': 'never'})
            self.assertEqual(target.read_text(), 'unchanged')
