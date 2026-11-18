import copy
import tempfile
import unittest
from pathlib import Path
from sol_pi_hermes.observation_pack import project_messages


class RequestCountTests(unittest.TestCase):
    def test_duplicate_observations_get_two_full_requests(self):
        for call_id in ('same', ''):
            with self.subTest(call_id=call_id), tempfile.TemporaryDirectory() as tmp:
                msg = {'role': 'tool', 'tool_call_id': call_id, 'content': 'x' * 15000}
                history = [msg, copy.deepcopy(msg), copy.deepcopy(msg)]
                original = copy.deepcopy(history)
                counts = {}
                for send in range(1, 4):
                    result = project_messages(history, Path(tmp), counts)
                    self.assertEqual(list(counts.values()), [send])
                    for item in result:
                        self.assertEqual(item['content'] == msg['content'], send <= 2)
                self.assertEqual(history, original)

    def test_distinct_call_ids_keep_independent_send_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = {'role': 'tool', 'tool_call_id': 'one', 'content': 'x' * 15000}
            second = {**first, 'tool_call_id': 'two'}
            counts = {}
            project_messages([first], Path(tmp), counts)
            project_messages([first], Path(tmp), counts)
            result = project_messages([first, second, second], Path(tmp), counts)
            self.assertNotEqual(result[0]['content'], first['content'])
            self.assertEqual(result[1:], [second, second])
            self.assertEqual(sorted(counts.values()), [1, 3])
