import json
from pathlib import Path
import tempfile
import unittest
from sol_pi_hermes.observation_pack import project_messages

class FailedObservationTests(unittest.TestCase):
    def test_explicit_failure_envelopes_never_pack_or_archive(self):
        for failure in ({'exit_code':1}, {'exit_code':124}, {'exit_code':-9}, {'error':'denied'}, {'success':False}):
            for blocks in (False,True):
                with self.subTest(failure=failure,blocks=blocks), tempfile.TemporaryDirectory() as tmp:
                    raw=json.dumps({'output':'diagnostic line\n'*1500,**failure})
                    content=[{'type':'text','text':raw}] if blocks else raw
                    source=[{'role':'tool','name':'terminal','content':content},{'role':'assistant'},{'role':'assistant'}]
                    counts={};root=Path(tmp)
                    for _ in range(3):self.assertEqual(project_messages(source,root,counts),source)
                    self.assertEqual(counts,{})
                    self.assertEqual(list(root.iterdir()),[])

    def test_success_and_unstructured_controls_still_pack(self):
        payloads=[{'exit_code':0,'error':None}, {'success':True}, {'nested':{'error':'a data field'}}]
        texts=[json.dumps({'output':'data line\n'*1500,**p}) for p in payloads]+['plain data\n'*1500, json.dumps(['error','data line\n'*1500])]
        for text in texts:
            with self.subTest(prefix=text[:50]),tempfile.TemporaryDirectory() as tmp:
                source=[{'role':'tool','content':text},{'role':'assistant'},{'role':'assistant'}]
                self.assertNotEqual(project_messages(source,Path(tmp),{}),source)
