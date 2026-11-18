import json
from pathlib import Path
import tempfile
import unittest
from sol_pi_hermes.observation_pack import RECALL_MAX_BYTES,create_observation,ensure_stored,recall

class RecallWireBudgetTests(unittest.TestCase):
    def test_serialized_pages_fit_and_reconstruct_exactly(self):
        for text in ('😀'*10000, '\x00'*20000, '\\"'*12000, 'plain ASCII '*3000):
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
