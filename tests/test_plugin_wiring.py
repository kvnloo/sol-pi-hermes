"""register(ctx) end to end: llm_request middleware projects, obs_recall reads it back."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.plugin import register


class Ctx:
    def __init__(self):
        self.tools, self.middleware = {}, {}

    def register_tool(self, **kw):
        self.tools[kw['name']] = kw['handler']

    def register_middleware(self, kind, fn):
        self.middleware[kind] = fn


class PluginWiringTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        patcher = patch('sol_pi_hermes.config._hermes_home', return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.content = 'synthetic observation\n' * 1200
        self.request = {'model': 'm', 'messages': [
            {'role': 'tool', 'tool_call_id': 'c1', 'content': self.content},
            {'role': 'assistant', 'content': '1'}, {'role': 'assistant', 'content': '2'}]}

    def test_enabled_middleware_projects_and_recall_is_session_scoped(self):
        ctx = Ctx()
        register(ctx, SolPiConfig(observation_pack=True))
        out = ctx.middleware['llm_request'](request=self.request, session_id='s1')
        self.assertEqual(out['source'], 'sol-pi')
        projected = out['request']['messages'][0]
        self.assertNotEqual(projected['content'], self.content)
        self.assertEqual(out['request']['model'], 'm')
        self.assertEqual(self.request['messages'][0]['content'], self.content)  # input untouched
        obs_id = re.search(r'id: (obs_\w+)', projected['content']).group(1)
        got = json.loads(ctx.tools['obs_recall']({'id': obs_id, 'offset': 0}, session_id='s1'))
        self.assertIn('synthetic observation', got['content'])
        chunks, offset = [], 0
        for _ in range(20):
            page = json.loads(ctx.tools['obs_recall']({'id': obs_id, 'offset': offset}, session_id='s1'))
            chunks.append(page['content'].split('\n', 2)[2])
            if page['eof']:
                break
            offset = page['next_offset']
        else:
            self.fail('recall never reached eof')
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(chunks), self.content)
        other = json.loads(ctx.tools['obs_recall']({'id': obs_id}, session_id='s2'))
        self.assertNotIn('synthetic observation', other.get('content', ''))

    def test_send_counts_persist_across_requests(self):
        ctx = Ctx()
        register(ctx, SolPiConfig(observation_pack=True))
        request = {'messages': [self.request['messages'][0]]}
        results = [ctx.middleware['llm_request'](request=request, session_id='s1') for _ in range(3)]
        self.assertEqual([r is not None for r in results], [False, False, True])

    def test_disabled_or_malformed_requests_pass_through(self):
        ctx = Ctx()
        register(ctx, SolPiConfig())
        self.assertIsNone(ctx.middleware['llm_request'](request=self.request, session_id='s1'))
        self.assertFalse((self.home / 'sol-pi').exists())
        ctx = Ctx()
        register(ctx, SolPiConfig(observation_pack=True))
        for bad in (None, 'x', [], {'messages': []}):
            self.assertIsNone(ctx.middleware['llm_request'](request=bad, session_id='s1'))


if __name__ == '__main__':
    unittest.main()
