import copy
import json
import unittest
from unittest.mock import patch
from sol_pi_hermes import observation_eval as e

class RawControlTests(unittest.TestCase):
    def fixture(self, control):
        return {'name':'raw','capture':{'elements':[]},'control':control,'tool_call_id':'offline'}

    def test_exact_formatting_string_shape_and_real_recall(self):
        raw=' { "label" : "caf\\u00e9 / café", "n": 1e0, "note" : "'+'x'*12000+'" }\n'
        self.assertNotEqual(raw,json.dumps(json.loads(raw)))
        f=self.fixture({'raw_text_payload':raw}); before=copy.deepcopy(f)
        seen=[]; real=e.project_request
        def observe(request,*args):
            seen.append(request['messages'][0]['content'])
            return real(request,*args)
        with patch.object(e,'project_request',observe):result=e.evaluate_fixture(f)
        self.assertEqual(seen,[raw]*3)
        self.assertTrue(all(isinstance(x,str) for x in seen))
        self.assertEqual(f,before)
        self.assertEqual([s['before_bytes'] for s in result['sends']],[len(raw.encode())]*3)
        self.assertEqual([s['packed'] for s in result['sends']],[False,False,True])
        self.assertTrue(result['sends'][2]['byte_exact_recall'])
        self.assertEqual(result['sends'][2]['lost_fields_after_recall'],[])

    def test_legacy_dict_and_blocks_unchanged(self):
        value={'label':'café'}; blocks=[{'type':'text','text':'original'}]
        self.assertEqual(e._control_content(self.fixture({'text_payload':value})),json.dumps(value))
        got=e._control_content(self.fixture({'content_blocks':blocks}))
        self.assertEqual(got,blocks);self.assertIsNot(got,blocks)

    def test_raw_validation_and_exclusive_forms(self):
        controls=[{}, {'raw_text_payload':None}, {'raw_text_payload':{}},
                  {'raw_text_payload':'[]'}, {'raw_text_payload':'not JSON'},
                  {'raw_text_payload':'"text"'}, {'raw_text_payload':'{"a":"\ud800"}'}]
        for other,value in [('text_payload',{}),('content_blocks',[])]:
            controls.append({'raw_text_payload':'{}',other:value})
        for c in controls:
            with self.subTest(control=repr(c)),self.assertRaises(ValueError):e._control_content(self.fixture(c))
        self.assertEqual(e._control_content(self.fixture({'raw_text_payload':' {} \n'})),' {} \n')
