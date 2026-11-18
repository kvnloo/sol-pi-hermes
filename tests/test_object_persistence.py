import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sol_pi_hermes.observation_pack import create_observation, ensure_stored, project_messages, recall


class ObjectPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.message = {'role': 'tool', 'tool_call_id': 'archive', 'content': 'synthetic observation\n' * 1200}
        self.observation = create_observation(self.message, self.root)

    def test_short_writes_preserve_full_recall(self):
        real_write = os.write
        def short_write(fd, data):
            return real_write(fd, data[:17])
        with patch('sol_pi_hermes.observation_pack.os.write', side_effect=short_write):
            projected = project_messages([self.message, {'role': 'assistant', 'content': '1'},
                                          {'role': 'assistant', 'content': '2'}], self.root, {})
        self.assertNotEqual(projected[0]['content'], self.message['content'])
        self.assertEqual(self.observation.file_path.read_bytes(), self.message['content'].encode())
        chunks, offset = [], 0
        while True:
            result = json.loads(recall(self.root, self.observation.id, offset))
            chunks.append(result['content'].split('\n', 2)[2])
            if result['eof']:
                break
            offset = result['next_offset']
        self.assertEqual(''.join(chunks), self.message['content'])

    def test_zero_progress_keeps_full_observation_and_no_object(self):
        with patch('sol_pi_hermes.observation_pack.os.write', return_value=0):
            projected = project_messages([self.message, {'role': 'assistant', 'content': '1'},
                                          {'role': 'assistant', 'content': '2'}], self.root, {})
        self.assertEqual(projected[0], self.message)
        self.assertFalse(self.observation.file_path.exists())
        self.assertEqual(list(self.observation.file_path.parent.iterdir()), [])

    def test_failed_write_leaves_no_partial_object_and_can_retry(self):
        real_write = os.write
        calls = 0
        def failed_write(fd, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(fd, data[:17])
            raise OSError('simulated full device')
        with patch('sol_pi_hermes.observation_pack.os.write', side_effect=failed_write):
            with self.assertRaises(OSError):
                ensure_stored(self.observation)
        self.assertFalse(self.observation.file_path.exists())
        self.assertEqual(list(self.observation.file_path.parent.iterdir()), [])
        ensure_stored(self.observation)
        self.assertEqual(self.observation.file_path.read_text(), self.message['content'])

    def test_existing_valid_object_is_not_replaced(self):
        ensure_stored(self.observation)
        before = self.observation.file_path.stat().st_ino
        ensure_stored(self.observation)
        self.assertEqual(before, self.observation.file_path.stat().st_ino)
        self.assertEqual(self.observation.file_path.stat().st_mode & 0o777, 0o600)

    def test_existing_corruption_is_not_overwritten(self):
        self.observation.file_path.parent.mkdir(parents=True)
        self.observation.file_path.write_text('broken')
        with self.assertRaises(OSError):
            ensure_stored(self.observation)
        self.assertEqual(self.observation.file_path.read_text(), 'broken')

    def test_object_is_not_visible_until_complete(self):
        real_write = os.write
        def inspect_write(fd, data):
            self.assertFalse(self.observation.file_path.exists())
            return real_write(fd, data[:1024])
        with patch('sol_pi_hermes.observation_pack.os.write', side_effect=inspect_write):
            ensure_stored(self.observation)
        self.assertEqual(self.observation.file_path.read_text(), self.message['content'])

    def test_unsupported_publication_fails_open_without_residue(self):
        with patch('sol_pi_hermes.observation_pack.os.link', side_effect=OSError('unsupported')):
            projected = project_messages([self.message, {'role': 'assistant', 'content': '1'},
                                          {'role': 'assistant', 'content': '2'}], self.root, {})
        self.assertEqual(projected[0], self.message)
        self.assertEqual(list(self.observation.file_path.parent.iterdir()), [])

    def test_concurrent_writers_publish_one_complete_object(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as workers:
            list(workers.map(lambda _: ensure_stored(self.observation), range(24)))
        self.assertEqual(self.observation.file_path.read_text(), self.message['content'])
        self.assertEqual(list(self.observation.file_path.parent.iterdir()), [self.observation.file_path])

    def test_equal_size_corruption_is_rejected_by_hash(self):
        self.observation.file_path.parent.mkdir(parents=True)
        corrupt = b'x' * self.observation.bytes
        self.observation.file_path.write_bytes(corrupt)
        with self.assertRaisesRegex(OSError, 'hash mismatch'):
            ensure_stored(self.observation)
        self.assertEqual(self.observation.file_path.read_bytes(), corrupt)
        self.assertEqual(len(list(self.observation.file_path.parent.iterdir())), 1)

    def test_symlink_at_object_path_is_rejected_and_target_untouched(self):
        self.observation.file_path.parent.mkdir(parents=True)
        target = self.root / 'target.txt'
        target.write_text(self.message['content'])
        self.observation.file_path.symlink_to(target)
        with self.assertRaisesRegex(OSError, 'not a regular file'):
            ensure_stored(self.observation)
        self.assertEqual(target.read_text(), self.message['content'])
        self.assertTrue(self.observation.file_path.is_symlink())
