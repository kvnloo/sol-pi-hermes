import unittest

from sol_pi_hermes.observation_pack import _complete_line_excerpt


class CompleteLineExcerptWholeText(unittest.TestCase):
    def test_text_fitting_budget_is_returned_whole_from_either_end(self):
        for text in ('a\nb\n', 'a\nb', 'a', '', '\n'):
            for from_end in (False, True):
                with self.subTest(text=text, from_end=from_end):
                    self.assertEqual(_complete_line_excerpt(text, 100, from_end), text)

    def test_unterminated_last_line_is_a_complete_line_at_the_tail(self):
        self.assertEqual(_complete_line_excerpt('aa\nbb', 2, True), 'bb')
        self.assertEqual(_complete_line_excerpt('aa\nbb', 5, True), 'aa\nbb')
        self.assertEqual(_complete_line_excerpt('aa\nbb', 3, False), 'aa\n')


if __name__ == '__main__':
    unittest.main()
