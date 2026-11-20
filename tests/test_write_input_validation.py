import json
import unittest
from sol_pi_hermes.config import SolPiConfig
from sol_pi_hermes.plugin import register

class Context:
    def __init__(self):self.tools={};self.calls=[]
    def register_tool(self,**kw):self.tools[kw['name']]=kw['handler']
    def register_middleware(self,*args):pass
    def dispatch_tool(self,name,args,**kw):
        self.calls.append((name,args,kw))
        return json.dumps({'bytes_written':len(args.get('content',''))} if name=='write_file' else {'exit_code':0})

class WriteInputValidationTests(unittest.TestCase):
    def test_invalid_write_arguments_never_dispatch(self):
        invalid=[{}, {'path':'target'}, {'path':'target','content':None},
                 {'path':'target','content':False}, {'path':'target','content':0},
                 {'path':'target','content':[]}, {'path':'target','content':{}},
                 {'content':'x'}, {'path':None,'content':'x'}, {'path':'','content':'x'},
                 {'path':5,'content':'x'}, {'path':['target'],'content':'x'}]
        for args in invalid:
            with self.subTest(args=args):
                ctx=Context();register(ctx,SolPiConfig(action_fusion=True))
                result=json.loads(ctx.tools['sol_pi_write']({**args,'then_run':{'command':'never'}},task_id='task'))
                self.assertEqual(ctx.calls,[])
                self.assertIn('error',result)

    def test_explicit_empty_content_remains_valid(self):
        ctx=Context();register(ctx,SolPiConfig(action_fusion=True))
        result=json.loads(ctx.tools['sol_pi_write']({'path':'target','content':'','then_run':{'command':'synthetic'}},task_id='task'))
        self.assertEqual([c[0] for c in ctx.calls],['write_file','terminal'])
        self.assertEqual(ctx.calls[0][1],{'path':'target','content':''})
        self.assertEqual(result['then_run'],'[then_run:succeeded]')

    def test_valid_text_preserved(self):
        ctx=Context();register(ctx,SolPiConfig(action_fusion=True))
        text='0\nUnicode 😀\n'
        ctx.tools['sol_pi_write']({'path':'target','content':text},session_id='session')
        self.assertEqual(ctx.calls,[('write_file',{'path':'target','content':text},{'session_id':'session'})])

    def test_non_object_arguments_are_rejected(self):
        for args in (None,[],False,'text'):
            ctx=Context();register(ctx,SolPiConfig(action_fusion=True))
            result=json.loads(ctx.tools['sol_pi_write'](args))
            self.assertIn('error',result);self.assertEqual(ctx.calls,[])
