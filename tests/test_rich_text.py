import copy
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.observation_pack import project_messages
from sol_pi_hermes.plugin import register


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

    def test_non_text_typed_blocks_are_not_packed(self):
        for block in [{'type': 'image', 'text': 'alt'}, {'type': 'input_text', 'text': 'x'},
                      {'type': None, 'text': 'x'}]:
            with self.subTest(block=block), tempfile.TemporaryDirectory() as tmp:
                message = {'role': 'tool', 'content': ['x' * 16000, block]}
                counts = {}
                for _ in range(3):
                    self.assertEqual(project_messages([message], Path(tmp), counts), [message])
                self.assertEqual(counts, {})
                self.assertEqual(list(Path(tmp).rglob('*')), [])


class RegisteredMiddlewareTests(unittest.TestCase):
    def test_llm_request_handler_packs_plain_but_keeps_rich(self):
        rich = {'role': 'tool', 'content': [{'type': 'text', 'text': 'y' * 16000, 'citations': [1]}]}
        plain = {'role': 'tool', 'content': 'x' * 16000}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('sol_pi_hermes.config._hermes_home', return_value=Path(tmp)):
            ctx = MagicMock()
            register(ctx, SolPiConfig(observation_pack=True))
            handler = ctx.register_middleware.call_args.args[1]
            for _ in range(2):  # positive control: plain result packs on the third send
                self.assertIsNone(handler(request={'messages': [plain]}, session_id='plain'))
            packed = handler(request={'messages': [plain]}, session_id='plain')
            self.assertIn('[large tool result replaced', packed['request']['messages'][0]['content'])
            self.assertTrue((Path(tmp) / 'sol-pi' / 'plain').is_dir())
            before = sorted(Path(tmp).rglob('*'))
            for _ in range(3):
                self.assertIsNone(handler(request={'messages': [rich]}, session_id='rich'))
            self.assertEqual(sorted(Path(tmp).rglob('*')), before)
