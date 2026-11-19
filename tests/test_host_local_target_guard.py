import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sol_pi_hermes.action_fusion import THEN_RUN_SUCCEEDED, execute_mutation_then_run


class HostLocalTargetGuardTests(unittest.TestCase):
    def run_fused(self, path):
        calls = []
        result = json.loads(execute_mutation_then_run(
            mutate=lambda: '{"success": true}', absolute_path=path, then_run={'command': 'c'},
            run_command=lambda *args: calls.append(args) or '{"exit_code": 0}'))
        return result, calls

    def test_missing_host_target_never_dispatches_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, calls = self.run_fused(Path(tmp) / 'gone')
        self.assertEqual(calls, [])

    def test_unchanged_host_target_dispatches_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'f'
            target.write_text('x')
            result, calls = self.run_fused(target)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['then_run'], THEN_RUN_SUCCEEDED)

    def test_changed_host_target_never_dispatches_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'f'
            target.write_text('x')
            with patch('sol_pi_hermes.action_fusion.file_sha256', side_effect=['a', 'b']):
                _, calls = self.run_fused(target)
        self.assertEqual(calls, [])
