import tempfile
import unittest
from pathlib import Path
from sol_pi_hermes.observation_pack import read_recall_chunk


def chunk(data, offset=0, max_bytes=100, max_lines=100):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, 'object')
        path.write_bytes(data)
        return read_recall_chunk(path, offset, max_bytes=max_bytes, max_lines=max_lines)


class RecallPagingTests(unittest.TestCase):
    def test_max_lines_cuts_after_the_nth_newline(self):
        got = chunk(b'a\nb\nc\nd\n', max_lines=2)
        self.assertEqual((got['text'], got['nextOffset'], got['eof']), ('a\nb\n', 4, False))

    def test_max_lines_one_keeps_first_line_including_newline(self):
        got = chunk(b'abc\ndef', max_lines=1)
        self.assertEqual((got['text'], got['nextOffset']), ('abc\n', 4))

    def test_byte_budget_cuts_exactly_at_max_bytes(self):
        got = chunk(b'abcdefghij', max_bytes=4)
        self.assertEqual((got['text'], got['nextOffset'], got['eof']), ('abcd', 4, False))

    def test_multibyte_char_straddling_budget_is_not_split_or_dropped(self):
        data = 'ab😀cd'.encode()  # emoji occupies bytes 2..5
        first = chunk(data, max_bytes=4)
        self.assertEqual((first['text'], first['nextOffset']), ('ab', 2))
        second = chunk(data, first['nextOffset'], max_bytes=4)
        self.assertEqual((second['text'], second['nextOffset']), ('😀', 6))

    def test_char_ending_exactly_at_budget_is_kept(self):
        got = chunk('ab😀cd'.encode(), max_bytes=6)
        self.assertEqual((got['text'], got['nextOffset']), ('ab😀', 6))

    def test_mid_character_offset_with_tiny_budget_reports_no_progress(self):
        # Offset 1 is inside the emoji; backoff reaches 0 and must stop there.
        # With `end >= 0` it would slice buffer[:-1] and surface a
        # UnicodeDecodeError (also a ValueError), so only the message differs.
        with self.assertRaisesRegex(ValueError, 'no progress'):
            chunk('😀'.encode(), offset=1, max_bytes=1)

    def test_pages_reassemble_losslessly_and_report_eof(self):
        data = 'α\nβγ\n😀x\nlast'.encode()
        out, offset = '', 0
        for _ in range(50):
            got = chunk(data, offset, max_bytes=5, max_lines=2)
            out += got['text']
            offset = got['nextOffset']
            if got['eof']:
                break
        self.assertEqual(out.encode(), data)
        self.assertEqual(offset, len(data))


if __name__ == '__main__':
    unittest.main()
