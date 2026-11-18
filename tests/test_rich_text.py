import copy
import tempfile
import unittest
from pathlib import Path
from sol_pi_hermes.observation_pack import project_messages


class RichTextPreservationTests(unittest.TestCase):
    def test_metadata_blocks_remain_intact_without_archive_or_counts(self):
        for key, value in [('annotations', [{'url': 'https://example.invalid/source'}]),
                           ('citations', [{'start_index': 0}]), ('id', 'block1'),
                           ('cache_control', {'type': 'ephemeral'}), ('future_field', False)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                message = {'role': 'tool', 'content': [
                    {'type': 'text', 'text': 'x' * 16000},
                    {'type': 'text', 'text': 'evidence', key: value}]}
                original = copy.deepcopy(message)
                counts = {}
                for _ in range(3):
                    self.assertEqual(project_messages([message], Path(tmp), counts), [original])
                self.assertEqual(counts, {})
                self.assertEqual(list(Path(tmp).rglob('*')), [])
                self.assertEqual(message, original)

    def test_malformed_text_blocks_are_not_coerced_or_packed(self):
        for block in [{'type': 'text'}, {'type': 'text', 'text': None},
                      {'type': 'text', 'text': False}, {'type': 'text', 'text': ['evidence']}]:
            with self.subTest(block=block), tempfile.TemporaryDirectory() as tmp:
                message = {'role': 'tool', 'content': ['x' * 16000, block]}
                counts = {}
                for _ in range(3):
                    self.assertEqual(project_messages([message], Path(tmp), counts), [message])
                self.assertEqual(counts, {})
                self.assertEqual(list(Path(tmp).rglob('*')), [])

    def test_plain_text_lists_still_pack(self):
        for content in [['x' * 16000], [{'type': 'text', 'text': 'x' * 16000}],
                        ['x' * 8000, {'type': 'text', 'text': 'y' * 8000}]]:
            with self.subTest(content_type=type(content[0])), tempfile.TemporaryDirectory() as tmp:
                message = {'role': 'tool', 'content': content}
                original = copy.deepcopy(message)
                counts = {}
                for _ in range(2):
                    self.assertEqual(project_messages([message], Path(tmp), counts), [original])
                result = project_messages([message], Path(tmp), counts)
                self.assertIn('[large tool result replaced', result[0]['content'][0]['text'])
                self.assertEqual(message, original)
