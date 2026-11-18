import errno
import fcntl
import json
import os
import stat
import subprocess
import sys
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
        for payload in ({'error': 'denied'}, {'success': False}, {'exit_code': 1}):
            with self.subTest(payload=payload):
                result = json.loads(execute_mutation_then_run(mutate=lambda: '{}', absolute_path=None,
                    then_run={'command': 'fixture'}, run_command=lambda *_: json.dumps(payload)))
                self.assertEqual(result['then_run'], '[then_run:failed]')

    def test_background_or_incomplete_result_is_unknown(self):
        for payload in ({'session_id': 'job', 'status': 'running'}, {'exit_code': None}, {}, None, [], 'plain output', {'exit_code': False}):
            with self.subTest(payload=payload):
                result = json.loads(execute_mutation_then_run(mutate=lambda: '{}', absolute_path=None,
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

    def test_ledger_o_nofollow_blocks_symlink_when_precheck_is_raced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target'
            target.write_text('unchanged')
            path = ledger_path(root)
            path.parent.mkdir()
            path.symlink_to(target)
            with patch.object(Path, 'is_symlink', return_value=False), self.assertRaises(OSError):
                append_ledger(root, {'event': 'never'})
            self.assertEqual(target.read_text(), 'unchanged')

    def test_ledger_append_after_torn_tail_keeps_new_record_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = ledger_path(root)
            path.parent.mkdir()
            torn = b'{"event": "ok"}\n{"event": "tor'
            path.write_bytes(torn)
            append_ledger(root, {'event': 'after'})
            data = path.read_bytes()
            self.assertTrue(data.startswith(torn + b'\n'))
            self.assertEqual(json.loads(data.splitlines()[-1])['event'], 'after')
            append_ledger(root, {'event': 'healthy'})
            self.assertEqual(path.read_bytes().count(b'\n\n'), 0)

    def test_ledger_record_bytes_are_unchanged_json_dumps_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.datetime') as clock:
                clock.now.return_value.isoformat.return_value = 'T'
                append_ledger(root, {'event': 'é', 'n': 1})
            self.assertEqual(ledger_path(root).read_bytes(),
                             b'{"timestamp": "T", "event": "\\u00e9", "n": 1}\n')

    def test_ledger_short_writes_complete_one_record(self):
        real_write = os.write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.os.write', lambda fd, b: real_write(fd, bytes(b[:7]))):
                append_ledger(root, {'event': 'short', 'blob': 'z' * 100})
            self.assertEqual(json.loads(ledger_path(root).read_text())['blob'], 'z' * 100)

    def test_ledger_lock_errors_and_contention_skip_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.fcntl.flock', side_effect=OSError(errno.ENOLCK, 'No locks available')):
                append_ledger(root, {'event': 'nolock'})
            with open(ledger_path(root), 'rb') as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                append_ledger(root, {'event': 'contended'})
            events = [json.loads(x)['event'] for x in ledger_path(root).read_text().splitlines()]
            self.assertEqual(events, [])

    def test_ledger_concurrent_writers_never_interleave(self):
        # Regression guard: tearing was not reproducible on Linux before the fix.
        with tempfile.TemporaryDirectory() as tmp:
            code = ('import sys; from pathlib import Path; from sol_pi_hermes.observation_pack import append_ledger\n'
                    'for i in range(20): append_ledger(Path(sys.argv[1]), {"w": sys.argv[2], "blob": sys.argv[2] * 20000})')
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
            procs = [subprocess.Popen([sys.executable, '-c', code, tmp, str(n)], env=env) for n in range(6)]
            for proc in procs:
                self.assertEqual(proc.wait(timeout=60), 0)
            records = [json.loads(line) for line in ledger_path(Path(tmp)).read_bytes().splitlines()]
            self.assertEqual(len(records), 120)
            self.assertTrue(all(r['blob'] == r['w'] * 20000 for r in records))

    def test_ledger_append_holds_flock_while_writing(self):
        real_write = os.write
        held = []

        def probe(fd, data):
            with open(ledger_path(root), 'rb') as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held.append(False)
                except BlockingIOError:
                    held.append(True)
            return real_write(fd, data)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.os.write', probe):
                append_ledger(root, {'event': 'locked'})
        self.assertEqual(held, [True])

    def test_ledger_append_without_fcntl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.fcntl', None):
                append_ledger(root, {'event': 'nofcntl'})
            self.assertEqual(ledger_path(root).read_bytes(), b'')
