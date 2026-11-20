import json
import unittest
from unittest.mock import Mock
from sol_pi_hermes.action_fusion import THEN_RUN_SKIPPED, THEN_RUN_SUCCEEDED, execute_mutation_then_run


class CommandNulTests(unittest.TestCase):
    def test_nul_command_rejected_before_any_side_effect(self):
        for command in ('\0', 'printf ok\0', '\0printf ok', 'printf\0 ok'):
            with self.subTest(command=repr(command)):
                mutate = Mock(return_value=json.dumps({'success': True}))
                run = Mock(return_value=json.dumps({'exit_code': 0}))
                result = json.loads(execute_mutation_then_run(
                    mutate=mutate, absolute_path=None, then_run={'command': command}, run_command=run))
                self.assertEqual(result['then_run'], THEN_RUN_SKIPPED)
                self.assertIn('error', result)
                mutate.assert_not_called()
                run.assert_not_called()

    def test_other_command_characters_forwarded_exactly(self):
        for command in ('printf "λ"\n', 'printf "\\0"', 'echo one\necho two'):
            with self.subTest(command=command):
                mutate = Mock(return_value=json.dumps({'success': True}))
                run = Mock(return_value=json.dumps({'exit_code': 0}))
                result = json.loads(execute_mutation_then_run(
                    mutate=mutate, absolute_path=None, then_run={'command': command}, run_command=run))
                self.assertEqual(result['then_run'], THEN_RUN_SUCCEEDED)
                mutate.assert_called_once_with()
                run.assert_called_once_with(command, None)
