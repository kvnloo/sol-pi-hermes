import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sol_pi_hermes.observation_pack import create_observation, ensure_stored, read_recall_chunk, recall


class RecallProgressTests(unittest.TestCase):
    def test_truncation_after_size_check_returns_error_not_stalled_page(self):
        for offset in (0, 100):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                obs = create_observation({'role': 'tool', 'content': 'x' * 20000}, root)
                ensure_stored(obs)
                real_fdopen = os.fdopen
                def truncate(*args, **kwargs):
                    obs.file_path.write_bytes(b'')
                    return real_fdopen(*args, **kwargs)
                with patch('os.fdopen', side_effect=truncate):
                    result = json.loads(recall(root, obs.id, offset))
                self.assertIn('error', result)
                self.assertNotIn('next_offset', result)
                self.assertFalse((root / 'observation-pack' / 'ledger.jsonl').exists())

    def test_utf8_budget_that_cannot_advance_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'object')
            path.write_text('😀text')
            with self.assertRaises(ValueError):
                read_recall_chunk(path, 0, max_bytes=3, max_lines=10)

    def test_empty_eof_pages_remain_successful(self):
        for body in (b'', b'abc'):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp, 'object')
                path.write_bytes(body)
                result = read_recall_chunk(path, len(body), max_bytes=100, max_lines=10)
                self.assertEqual(result['bytes'], 0)
                self.assertEqual(result['nextOffset'], len(body))
                self.assertTrue(result['eof'])
