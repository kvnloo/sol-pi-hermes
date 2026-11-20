import json
import unittest
from sol_pi_hermes.action_fusion import execute_mutation_then_run
from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.plugin import register


class MutationConfirmationTests(unittest.TestCase):
    def test_unconfirmed_mutation_never_dispatches_command(self):
        for raw in ('', 'not-json', 'null', '[]', 'true', '{}', '{"success":null}',
                    '{"success":1}', '{"success":"true"}', '{"status":"pending"}', '{"exit_code":0}', '{"bytes_written":-1}', '{"bytes_written":true}', '{"bytes_written":"12"}', '{"bytes_written":1.0}', '{"success":false,"bytes_written":12}'):
            with self.subTest(raw=raw):
                calls = []
                result = json.loads(execute_mutation_then_run(mutate=lambda:raw, absolute_path=None,
                    then_run={'command':'not-executed'}, run_command=lambda *args:calls.append(args) or '{"exit_code":0}'))
                self.assertEqual(calls, [])
                self.assertEqual(result['then_run'], '[then_run:skipped]')
                self.assertEqual(result['mutation'], raw)

    def test_mutate_exception_never_runs_followup_and_stays_json(self):
        def boom():
            raise OSError('disk full')
        for then_run in ({'command':'not-executed'}, None):
            with self.subTest(then_run=then_run):
                calls = []
                result = json.loads(execute_mutation_then_run(mutate=boom, absolute_path=None,
                    then_run=then_run, run_command=lambda *args:calls.append(args) or '{"exit_code":0}'))
                self.assertEqual(calls, [])
                self.assertIn('disk full', result['error'])
                self.assertEqual('then_run' in result or '[then_run:skipped]' in result['error'], then_run is not None)

    def test_confirmed_success_and_nochange_allow_followup(self):
        for payload in ({'success':True}, {'success':True,'no_change':True}, {'bytes_written':0}, {'bytes_written':23,'verified':True}):
            calls = []
            result = json.loads(execute_mutation_then_run(mutate=lambda:json.dumps(payload), absolute_path=None,
                then_run={'command':'synthetic'}, run_command=lambda *args:calls.append(args) or '{"exit_code":0}'))
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['then_run'], '[then_run:succeeded]')

    def test_non_true_success_field_is_not_rescued_by_bytes_written(self):
        for raw in ('{"success":null,"bytes_written":12}', '{"success":1,"bytes_written":12}',
                    '{"success":"true","bytes_written":12}'):
            with self.subTest(raw=raw):
                calls = []
                result = json.loads(execute_mutation_then_run(mutate=lambda:raw, absolute_path=None,
                    then_run={'command':'never'}, run_command=lambda *a:calls.append(a) or '{"exit_code":0}'))
                self.assertEqual(calls, [])
                self.assertEqual(result['then_run'], '[then_run:skipped]')

    def test_nonzero_exit_code_overrides_success(self):
        for code in (1, -1):
            with self.subTest(code=code):
                calls = []
                result = json.loads(execute_mutation_then_run(mutate=lambda:json.dumps({'success':True,'exit_code':code}), absolute_path=None,
                    then_run={'command':'never'}, run_command=lambda *a:calls.append(a) or '{}'))
                self.assertEqual(calls, [])
                self.assertEqual(result['then_run'], '[then_run:skipped]')

    def test_reported_error_overrides_success(self):
        result = json.loads(execute_mutation_then_run(mutate=lambda:'{"success":true,"error":"denied"}', absolute_path=None,
            then_run={'command':'never'}, run_command=lambda *_:self.fail('unexpected command')))
        self.assertEqual(result['then_run'], '[then_run:skipped]')

    def test_reported_failure_text_is_surfaced_not_generic(self):
        cases = [({'success':True,'exit_code':3}, 'Command exited with code 3'),
                 ({'bytes_written':3,'error':'disk full'}, 'disk full'),
                 ({'success':False,'message':'no match'}, 'no match')]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                result = json.loads(execute_mutation_then_run(mutate=lambda:json.dumps(payload), absolute_path=None,
                    then_run={'command':'never'}, run_command=lambda *_:self.fail('unexpected command')))
                self.assertEqual(result['error'], expected)
                self.assertEqual(result['then_run'], '[then_run:skipped]')

    def test_falsy_error_and_boolean_exit_code_are_not_failures(self):
        # Pins the "truthy error" and "exact int exit_code" contracts (null error is a plausible success shape).
        for out, status in (('{"exit_code":0,"error":null}', '[then_run:succeeded]'),
                            ('{"exit_code":0,"error":""}', '[then_run:succeeded]'),
                            ('{"exit_code":true}', '[then_run:unknown]')):
            with self.subTest(out=out):
                calls = []
                result = json.loads(execute_mutation_then_run(mutate=lambda:'{"success":true,"error":null}', absolute_path=None,
                    then_run={'command':'go'}, run_command=lambda *a:calls.append(a) or out))
                self.assertEqual(len(calls), 1)
                self.assertEqual(result['then_run'], status)

    def test_unconfirmed_result_without_failure_gets_generic_error(self):
        result = json.loads(execute_mutation_then_run(mutate=lambda:'{}', absolute_path=None,
            then_run={'command':'never'}, run_command=lambda *_:self.fail('unexpected command')))
        self.assertIn('not confirmed', result['error'])

    def test_no_followup_preserves_unknown_result(self):
        self.assertEqual(execute_mutation_then_run(mutate=lambda:'unknown', absolute_path=None,
            then_run=None, run_command=lambda *_:self.fail('unexpected command')), 'unknown')

    def test_registered_write_and_patch_stop_after_unknown_mutation(self):
        class Context:
            def __init__(self): self.tools = {}; self.calls = []
            def register_tool(self, **kw): self.tools[kw['name']] = kw['handler']
            def register_middleware(self, *args): pass
            def dispatch_tool(self, name, args, **kw): self.calls.append(name); return 'null'
        for name, expected in [('sol_pi_write','write_file'), ('sol_pi_patch','patch')]:
            ctx = Context(); register(ctx, SolPiConfig(action_fusion=True))
            result = json.loads(ctx.tools[name]({'path':'synthetic.txt','content':'x','then_run':{'command':'never'}},task_id='synthetic'))
            self.assertEqual(ctx.calls,[expected])
            self.assertEqual(result['then_run'], '[then_run:skipped]')

    def test_registered_patch_confirmed_forwards_args_without_then_run_then_runs_command(self):
        class Context:
            def __init__(self): self.tools = {}; self.calls = []
            def register_tool(self, **kw): self.tools[kw['name']] = kw['handler']
            def register_middleware(self, *args): pass
            def dispatch_tool(self, name, args, **kw):
                self.calls.append((name, args, kw))
                return '{"success": true}' if name == 'patch' else '{"exit_code": 0}'
        ctx = Context(); register(ctx, SolPiConfig(action_fusion=True))
        ctx.tools['sol_pi_patch']({'path': 'a.txt', 'old_string': 'x', 'new_string': 'y',
            'then_run': {'command': 'echo hi', 'timeout': 5}}, task_id='t1')
        self.assertEqual(ctx.calls, [
            ('patch', {'path': 'a.txt', 'old_string': 'x', 'new_string': 'y'}, {'task_id': 't1'}),
            ('terminal', {'command': 'echo hi', 'timeout': 5}, {'task_id': 't1'}),
        ])
