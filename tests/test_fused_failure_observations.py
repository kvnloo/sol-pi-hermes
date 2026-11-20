import json
from pathlib import Path
import tempfile
import unittest

from sol_pi_hermes.action_fusion import execute_mutation_then_run
from sol_pi_hermes.observation_pack import project_messages


class FusedFailureObservationTests(unittest.TestCase):
    def fused_result(self, result):
        return execute_mutation_then_run(
            mutate=lambda: '{"success":true}', absolute_path=None,
            then_run={"command": "synthetic callback only"},
            run_command=lambda *_: json.dumps({"output": "diagnostic line\n" * 1500, **result}),
        )

    def test_failed_fused_commands_remain_visible_without_archive_or_accounting(self):
        for failure in ({"exit_code": 1}, {"exit_code": -9}, {"error": "denied"}):
            for blocks in (False, True):
                with self.subTest(failure=failure, blocks=blocks), tempfile.TemporaryDirectory() as tmp:
                    raw = self.fused_result(failure)
                    self.assertEqual(json.loads(raw)["then_run"], "[then_run:failed]")
                    content = [{"type": "text", "text": raw}] if blocks else raw
                    source = [{"role": "tool", "name": "sol_pi_write", "content": content},
                              {"role": "assistant"}, {"role": "assistant"}]
                    counts = {}
                    for _ in range(3):
                        self.assertEqual(project_messages(source, Path(tmp), counts), source)
                    self.assertEqual(counts, {})
                    self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_success_unknown_and_nested_data_still_pack(self):
        texts = [self.fused_result({"exit_code": 0}), self.fused_result({"status": "running"}),
                 json.dumps({"data": {"then_run": "[then_run:failed]"}, "output": "data\n" * 3000})]
        for text in texts:
            with self.subTest(text=text[:60]), tempfile.TemporaryDirectory() as tmp:
                source = [{"role": "tool", "name": "sol_pi_write", "content": text},
                          {"role": "assistant"}, {"role": "assistant"}]
                self.assertNotEqual(project_messages(source, Path(tmp), {}), source)
