import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.observation_pack import create_observation, ensure_stored, runtime_root_for_session
from sol_pi_hermes.plugin import register


class Context:
    def __init__(self):
        self.tools, self.middleware = {}, {}
    def register_tool(self, **kwargs):
        self.tools[kwargs['name']] = kwargs['handler']
    def register_middleware(self, kind, callback):
        self.middleware[kind] = callback


class IdentityAdmissionTests(unittest.TestCase):
    def test_missing_identity_does_not_archive_or_project(self):
        ctx = Context()
        register(ctx, SolPiConfig(observation_pack=True))
        request = {'messages': [{'role': 'tool', 'content': 'private\n' * 3000}]}
        original = copy.deepcopy(request)
        with patch('sol_pi_hermes.plugin.runtime_root_for_session') as root:
            for scope in ({}, {'session_id': None}, {'session_id': '', 'task_id': ''},
                          {'session_id': 7}, {'session_id': [], 'task_id': False}):
                with self.subTest(scope=scope):
                    for _ in range(4):
                        self.assertIsNone(ctx.middleware['llm_request'](request=request, **scope))
            root.assert_not_called()
        self.assertEqual(request, original)

    def test_missing_identity_cannot_recall_explicit_default_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            observation = create_observation({'role': 'tool', 'content': 'secret\n' * 3000},
                                             runtime_root_for_session('default', home))
            ensure_stored(observation)
            ctx = Context()
            with patch('sol_pi_hermes.plugin.runtime_root_for_session',
                       side_effect=lambda sid: runtime_root_for_session(sid, home)):
                register(ctx, SolPiConfig(observation_pack=True))
                anonymous = json.loads(ctx.tools['obs_recall']({'id': observation.id}))
                owner = json.loads(ctx.tools['obs_recall']({'id': observation.id}, session_id='default'))
            self.assertIn('error', anonymous)
            self.assertNotIn('secret', json.dumps(anonymous))
            self.assertIn('secret', owner['content'])

    def test_session_precedence_and_valid_task_fallback(self):
        ctx = Context()
        register(ctx, SolPiConfig(observation_pack=True))
        request = {'messages': [{'role': 'tool', 'content': 'small'}]}
        with patch('sol_pi_hermes.plugin.project_request', return_value=None), \
             patch('sol_pi_hermes.plugin.runtime_root_for_session', return_value=Path('/unused')) as root:
            ctx.middleware['llm_request'](request=request, session_id='session', task_id='task')
            root.assert_called_once_with('session')
            root.reset_mock()
            ctx.middleware['llm_request'](request=request, session_id='', task_id='task')
            root.assert_called_once_with('task')
