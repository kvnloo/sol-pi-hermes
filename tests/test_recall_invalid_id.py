import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sol_pi_hermes.observation_pack import recall, RECALL_MAX_BYTES


class RecallInvalidIdTests(unittest.TestCase):
    def test_large_malformed_ids_have_bounded_nonreflecting_error(self):
        for obs_id in ('private-input-marker' * 2000, '😀\n' * 10000):
            with self.subTest(kind=obs_id[:4]), tempfile.TemporaryDirectory() as tmp:
                with patch('sol_pi_hermes.observation_pack.observation_path', side_effect=AssertionError('invalid ID reached IO')):
                    result = recall(Path(tmp), obs_id)
                self.assertIn('error', json.loads(result))
                self.assertLessEqual(len(result.encode()), RECALL_MAX_BYTES)
                self.assertNotIn(obs_id[:4], result)
                self.assertEqual(list(Path(tmp).rglob('*')), [])

    def test_nonstring_id_returns_error(self):
        for obs_id in (None, 123, [], {}):
            with self.subTest(obs_id=obs_id), tempfile.TemporaryDirectory() as tmp:
                self.assertIn('error', json.loads(recall(Path(tmp), obs_id)))

    def test_valid_unknown_id_remains_identifiable(self):
        obs_id = 'obs_' + 'a' * 24
        with tempfile.TemporaryDirectory() as tmp:
            result = json.loads(recall(Path(tmp), obs_id))
        self.assertEqual(result, {'error': f'Unknown observation id: {obs_id}'})
