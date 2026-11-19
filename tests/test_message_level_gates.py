import tempfile
import unittest
from pathlib import Path
from sol_pi_hermes.observation_pack import (
    THRESHOLD_BYTES, create_observation, is_pure_text_tool_result,
    project_messages, tool_call_id_of, tool_name_of)

BIG = 'data line\n' * 1500


def packed(message):
    """True if the third send of a big tool message is replaced by a placeholder."""
    source = [message, {'role': 'assistant'}, {'role': 'assistant'}]
    with tempfile.TemporaryDirectory() as tmp:
        counts, out = {}, None
        for _ in range(3):
            out = project_messages(source, Path(tmp), counts)
        return out != source


class MessageLevelGateTests(unittest.TestCase):
    def test_tool_roles_pack_as_positive_control(self):
        for role in ('tool', 'toolResult'):
            with self.subTest(role=role):
                self.assertTrue(packed({'role': role, 'content': BIG}))

    def test_non_tool_roles_never_pack(self):
        for role in ('user', 'assistant', 'system', None):
            with self.subTest(role=role):
                self.assertFalse(packed({'role': role, 'content': BIG}))

    def test_each_message_level_error_flag_alone_prevents_packing(self):
        for flags in ({'isError': True}, {'is_error': True}):
            with self.subTest(flags=flags):
                self.assertFalse(packed({'role': 'tool', 'content': BIG, **flags}))
        for flags in ({'isError': False}, {'is_error': False}, {'isError': False, 'is_error': False}):
            with self.subTest(flags=flags):
                self.assertTrue(packed({'role': 'tool', 'content': BIG, **flags}))

    def test_predicate_rejects_missing_content_and_accepts_text(self):
        self.assertFalse(is_pure_text_tool_result({'role': 'tool'}))
        self.assertFalse(is_pure_text_tool_result({'role': 'tool', 'content': None}))
        self.assertTrue(is_pure_text_tool_result({'role': 'tool', 'content': ''}))

    def test_name_and_id_fallback_order_and_defaults(self):
        self.assertEqual(tool_name_of({}), 'tool')
        for key in ('name', 'toolName', 'tool_name'):
            self.assertEqual(tool_name_of({key: 'x'}), 'x')
        self.assertEqual(tool_name_of({'name': 'a', 'toolName': 'b', 'tool_name': 'c'}), 'a')
        self.assertEqual(tool_name_of({'toolName': 'b', 'tool_name': 'c'}), 'b')
        self.assertEqual(tool_call_id_of({}), '')
        for key in ('tool_call_id', 'toolCallId', 'id'):
            self.assertEqual(tool_call_id_of({key: 'x'}), 'x')
        self.assertEqual(tool_call_id_of({'tool_call_id': 'a', 'toolCallId': 'b', 'id': 'c'}), 'a')
        self.assertEqual(tool_call_id_of({'toolCallId': 'b', 'id': 'c'}), 'b')


class SizeThresholdTests(unittest.TestCase):
    def test_exactly_threshold_bytes_stays_inline(self):
        with tempfile.TemporaryDirectory() as tmp:
            at = {'role': 'tool', 'content': 'x' * THRESHOLD_BYTES}
            over = {'role': 'tool', 'content': 'x' * (THRESHOLD_BYTES + 1)}
            self.assertIsNone(create_observation(at, Path(tmp)))
            self.assertEqual(create_observation(over, Path(tmp)).bytes, THRESHOLD_BYTES + 1)


if __name__ == '__main__':
    unittest.main()
