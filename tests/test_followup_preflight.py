import json
import unittest
from sol_pi_hermes.action_fusion import THEN_RUN_SKIPPED
from sol_pi_hermes.plugin import register
from sol_pi_hermes.config import SolPiConfig

class Context:
    def __init__(self):self.tools={};self.calls=[]
    def register_tool(self,**kw):self.tools[kw['name']]=kw['handler']
    def register_middleware(self,*args):pass
    def dispatch_tool(self,name,args,**kw):
        self.calls.append((name,args))
        return json.dumps({'exit_code':0} if name=='terminal' else {'success':True})

class FollowupPreflightTests(unittest.TestCase):
    def test_invalid_followup_refused_before_any_mutation(self):
        specs=[[], 'echo bad', False, {}, {'command':None}, {'command':1}, {'command':''}, {'command':'  '}]
        specs += [{'command':'synthetic','timeout':v} for v in (-1,False,'1',float('inf'),float('nan'),10**1000)]
        for name in ('sol_pi_write','sol_pi_patch'):
            for spec in specs:
                with self.subTest(name=name,spec=spec):
                    ctx=Context();register(ctx,SolPiConfig(action_fusion=True))
                    out=json.loads(ctx.tools[name]({'path':'target','content':'safe','then_run':spec}))
                    self.assertEqual(ctx.calls,[])
                    self.assertIn('error',out)
                    self.assertEqual(out['then_run'],THEN_RUN_SKIPPED)

    def test_omitted_or_null_followup_keeps_mutation_only(self):
        for extra in ({},{'then_run':None}):
            ctx=Context();register(ctx,SolPiConfig(action_fusion=True))
            ctx.tools['sol_pi_write']({'path':'target','content':'',**extra})
            self.assertEqual([c[0] for c in ctx.calls],['write_file'])

    def test_valid_followup_preserves_command_and_timeout(self):
        for timeout in (None,0,0.5,10):
            ctx=Context();register(ctx,SolPiConfig(action_fusion=True))
            spec={'command':' echo synthetic '}
            if timeout is not None:spec['timeout']=timeout
            result=json.loads(ctx.tools['sol_pi_write']({'path':'target','content':'','then_run':spec}))
            self.assertEqual(ctx.calls[-1],('terminal',spec))
            self.assertEqual(result['then_run'],'[then_run:succeeded]')


class FollowupSchemaTests(unittest.TestCase):
    def test_registered_timeout_schema_matches_preflight(self):
        schemas = {}
        ctx = Context()
        ctx.register_tool = lambda **kw: schemas.__setitem__(kw['name'], kw['schema'])
        register(ctx, SolPiConfig(action_fusion=True))
        for name in ('sol_pi_write', 'sol_pi_patch'):
            timeout = schemas[name]['parameters']['properties']['then_run']['properties']['timeout']
            self.assertEqual((timeout.get('type'), timeout.get('minimum')), ('number', 0))
