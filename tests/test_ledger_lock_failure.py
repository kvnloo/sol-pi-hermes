import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.observation_pack import (
    FULL_SENDS, THRESHOLD_BYTES, append_ledger, create_observation, ledger_path,
    project_request, recall,
)


class LedgerLockFailureTests(unittest.TestCase):
    def test_lock_retry_budget_matches_documented_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.fcntl.flock',
                       side_effect=BlockingIOError) as flock, \
                 patch('sol_pi_hermes.observation_pack.time.sleep') as sleep:
                append_ledger(root, {'event': 'busy'})
            self.assertEqual(flock.call_count, 50)
            self.assertEqual({c.args for c in sleep.call_args_list}, {(0.002,)})
            self.assertEqual(ledger_path(root).read_bytes(), b'')

    def test_lock_acquired_after_contention_writes_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.fcntl.flock',
                       side_effect=[BlockingIOError, BlockingIOError, None]) as flock, \
                 patch('sol_pi_hermes.observation_pack.time.sleep'):
                append_ledger(root, {'event': 'late'})
            self.assertEqual(flock.call_count, 3)
            self.assertEqual(json.loads(ledger_path(root).read_text())['event'], 'late')

    def test_unavailable_lock_never_starts_a_short_write(self):
        real_write = os.write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            append_ledger(root, {'event': 'existing'})
            before = ledger_path(root).read_bytes()
            interleaved = False

            def short_write(fd, data):
                nonlocal interleaved
                if not interleaved:
                    interleaved = True
                    n = real_write(fd, bytes(data[:7]))
                    append_ledger(root, {'event': 'other'})
                    return n
                return real_write(fd, data)

            with patch('sol_pi_hermes.observation_pack.fcntl.flock',
                       side_effect=OSError(errno.ENOLCK, 'No locks available')), \
                 patch('sol_pi_hermes.observation_pack.os.write', short_write):
                append_ledger(root, {'event': 'first'})
            self.assertEqual(ledger_path(root).read_bytes(), before)
            self.assertFalse(interleaved)

    def test_zero_byte_write_raises_instead_of_spinning(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The second mocked call raises to keep the old buggy loop bounded.
            with patch('sol_pi_hermes.observation_pack.os.write',
                       side_effect=[0, RuntimeError('would spin')]) as write:
                with self.assertRaises(OSError):
                    append_ledger(Path(tmp), {'event': 'zero'})
                self.assertEqual(write.call_count, 1)

    def test_projection_and_recall_survive_unavailable_lock(self):
        body = 'retained output' * THRESHOLD_BYTES
        message = {'role': 'tool', 'name': 'terminal', 'tool_call_id': 'x', 'content': body}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('sol_pi_hermes.observation_pack.fcntl', None):
                sent = {}
                for _ in range(FULL_SENDS + 1):
                    result = project_request({'messages': [message]}, root, sent)
                observation = create_observation(message, root)
                self.assertNotEqual(result['request']['messages'][0]['content'], body)
                recalled = json.loads(recall(root, observation.id, 0))
            self.assertIn('retained output', recalled['content'])
            self.assertEqual(message['content'], body)
            self.assertEqual(ledger_path(root).read_bytes(), b'')
