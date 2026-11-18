import json
import copy
from unittest import mock
from sol_pi_hermes import observation_eval
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from sol_pi_hermes.observation_eval import DEFAULT_FIXTURES, main


class RichTextShadowGateTests(unittest.TestCase):
    def run_case(self, content):
        fixtures = json.loads(DEFAULT_FIXTURES.read_text())
        fixture = fixtures['cases'][0]
        fixture['control'] = {'content_blocks': content}
        fixtures['cases'] = [fixture]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'fixtures.json')
            path.write_text(json.dumps(fixtures))
            with redirect_stdout(StringIO()) as output:
                code = main(['--fixtures', str(path)])
        return code, json.loads(output.getvalue())['cases'][0]

    def test_cli_rejects_lost_rich_metadata_despite_exact_text_recall(self):
        for key, value in [('annotations', [{'source': 'synthetic'}]),
                           ('citations', [{'id': 'cite1'}]), ('cache_control', {'type': 'ephemeral'}),
                           ('future_metadata', False)]:
            with self.subTest(key=key):
                real = observation_eval.project_request
                def lose_metadata(request, root, counts):
                    stripped = copy.deepcopy(request)
                    stripped['messages'][0]['content'] = [
                        {'type': 'text', 'text': 'x' * 16000}
                    ]
                    return real(stripped, root, counts)
                with mock.patch.object(observation_eval, 'project_request', lose_metadata):
                    code, case = self.run_case([{'type': 'text', 'text': 'x' * 16000, key: value}])
                self.assertTrue(case['sends'][2]['byte_exact_recall'])
                self.assertFalse(case['non_text_content_unchanged'])
                self.assertEqual(code, 1)

    def test_cli_allows_plain_string_blocks_with_exact_text_recall(self):
        code, case = self.run_case(['x' * 16000])
        self.assertTrue(case['sends'][2]['byte_exact_recall'])
        self.assertTrue(case['non_text_content_unchanged'])
        self.assertEqual(code, 0)
