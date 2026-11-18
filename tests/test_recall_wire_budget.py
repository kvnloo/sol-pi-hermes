import json
from pathlib import Path
import tempfile
import unittest
from sol_pi_hermes.observation_pack import RECALL_MAX_BYTES,create_observation,ensure_stored,ledger_path,recall

class RecallWireBudgetTests(unittest.TestCase):
    def test_serialized_pages_fit_and_reconstruct_exactly(self):
        for text in ('😀'*10000, '\x00'*20000, '\\"'*12000, 'plain ASCII '*3000, 'a\x01é\t€\x1f😀"\\'*3500):
            with self.subTest(text_prefix=repr(text[:10])), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                obs=create_observation({'role':'tool','content':text},root)
                ensure_stored(obs)
                offset=0; parts=[]
                for _ in range(100):
                    raw=recall(root,obs.id,offset)
                    self.assertLessEqual(len(raw.encode('utf-8')),RECALL_MAX_BYTES)
                    page=json.loads(raw)
                    self.assertNotIn('error',page)
                    chunk=page['content'].split('\n',2)[2]
                    self.assertEqual(page['bytes'],len(chunk.encode('utf-8')))
                    self.assertEqual(page['next_offset'],offset+page['bytes'])
                    parts.append(chunk)
                    if page['eof']:break
                    self.assertGreater(page['next_offset'],offset)
                    offset=page['next_offset']
                else:self.fail('pagination did not terminate')
                self.assertEqual(''.join(parts),text)

    def test_shrunk_page_writes_one_matching_ledger_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            obs=create_observation({'role':'tool','content':'\x00'*20000},root)
            ensure_stored(obs)
            page=json.loads(recall(root,obs.id,0))
            recs=[json.loads(l) for l in ledger_path(root).read_text().splitlines()]
            recs=[r for r in recs if r.get('event')=='recall']
            self.assertEqual(len(recs),1)
            r=recs[0]
            self.assertEqual((r['bytes'],r['nextOffset'],r['eof'],r['lines']),(page['bytes'],page['next_offset'],page['eof'],page['lines']))
            self.assertLess(page["bytes"],RECALL_MAX_BYTES//2)
