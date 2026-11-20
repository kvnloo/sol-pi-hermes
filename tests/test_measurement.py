import contextlib
import copy
import io
import json
import sys
import unittest
from unittest import mock

from benchmarks import observation_pack
from benchmarks.observation_pack import fixtures, main, measure


class MeasurementTests(unittest.TestCase):
    def test_large_rows_preserve_first_two_and_recall_exactly(self):
        for name in ('large_ascii', 'large_unicode'):
            with self.subTest(name=name):
                source = fixtures()[name]
                original = copy.deepcopy(source)
                result = measure(source)
                self.assertEqual(result['counter'], 'ceil_unicode_chars_div_4_estimate')
                self.assertEqual(source, original)
                self.assertTrue(result['transcript_unchanged'])
                self.assertTrue(result['recall_roundtrip'])
                self.assertEqual(result['recalled_observations'], 1)
                for row in result['requests'][:2]:
                    self.assertEqual(row['control_bytes'], row['projected_bytes'])
                self.assertLess(result['requests'][2]['projected_bytes'], result['requests'][2]['control_bytes'])
                self.assertEqual(result['requests'][2]['projected_bytes'], result['requests'][3]['projected_bytes'])
                self.assertEqual(result['net_saved_units_after_one_full_recall_response'],
                                 result['saved_units_before_recall'] - result['recall_response_units'])
                self.assertIsNone(result['provider_billed_tokens'])

    def test_small_error_and_multimodal_controls_are_not_savings(self):
        for name in ('small_control', 'error_control', 'mixed_image_control'):
            with self.subTest(name=name):
                result = measure(fixtures()[name])
                self.assertEqual(result['saved_bytes_before_recall'], 0)
                self.assertEqual(result['recall_response_units'], 0)

    def test_short_run_no_savings_or_recall(self):
        result = measure(fixtures()['large_ascii'], requests=2)
        self.assertEqual(result['saved_units_before_recall'], 0)
        self.assertEqual(result['recalled_observations'], 0)

    def test_custom_counter_and_duplicate_recall(self):
        source = fixtures()['large_unicode'] * 2
        result = measure(source, counter=lambda text: len(text.encode()), counter_name='bytes')
        self.assertEqual(result['counter'], 'bytes')
        self.assertEqual(result['control_units'], result['control_bytes'])
        self.assertEqual(result['recalled_observations'], 1)

    def test_unnamed_custom_counter_is_not_labeled_as_char_estimate(self):
        result = measure(fixtures()['small_control'], requests=1, counter=len)
        self.assertEqual(result['counter'], 'unnamed_custom_counter')

    def test_default_estimate_is_ceil_of_unicode_chars_over_four(self):
        source = [{'role': 'tool', 'name': 't', 'tool_call_id': 'x', 'content': '測定🙂a'}]
        text = observation_pack.wire({'messages': source})
        row = measure(source, requests=1)['requests'][0]
        self.assertNotEqual(len(text) % 4, 0)
        self.assertNotEqual(len(text), len(text.encode()))
        self.assertEqual(row['control_units'], -(-len(text) // 4))

    def test_recall_can_outweigh_gross_savings(self):
        result = measure(fixtures()['large_ascii'], requests=3)
        self.assertLess(result['net_saved_units_after_one_full_recall_response'], 0)

    def test_aggregate_totals_are_sums_of_request_rows(self):
        for name in ('large_ascii', 'large_unicode'):
            with self.subTest(name=name):
                result = measure(fixtures()[name])
                for key in ('control_bytes', 'projected_bytes', 'control_units', 'projected_units'):
                    self.assertEqual(result[key], sum(row[key] for row in result['requests']), key)
                self.assertLess(result['projected_bytes'], result['control_bytes'])
                self.assertLess(result['projected_units'], result['control_units'])

    def test_wire_serialization_is_pinned(self):
        self.assertEqual(observation_pack.wire({'b': '測', 'a': [1, None]}), '{"a":[1,null],"b":"測"}')

    def test_transcript_unchanged_flag_reports_mutation_after_projection(self):
        source = fixtures()['large_ascii']
        real = observation_pack.recall
        def mutate(*a):
            source[0]['content'] = 'tampered'
            return real(*a)
        with mock.patch.object(observation_pack, 'recall', mutate):
            self.assertFalse(measure(source)['transcript_unchanged'])

    def test_invalid_request_count(self):
        with self.assertRaises(ValueError):
            measure([], requests=0)

    def test_recall_accounting_covers_every_page(self):
        pages = []
        real = observation_pack.recall
        with mock.patch.object(observation_pack, 'recall',
                               lambda *a: pages.append(real(*a)) or pages[-1]):
            result = measure(fixtures()['large_ascii'])
        self.assertGreater(len(pages), 1)
        self.assertTrue(result['recall_roundtrip'])
        self.assertEqual(result['recall_response_bytes'],
                         sum(len(observation_pack.wire(json.loads(p)).encode()) for p in pages))

    def test_unicode_recall_bytes_are_utf8_and_duplicates_counted_once(self):
        pages = []
        real = observation_pack.recall
        with mock.patch.object(observation_pack, 'recall',
                               lambda *a: pages.append(real(*a)) or pages[-1]):
            once = measure(fixtures()['large_unicode'])
            pages_once = len(pages)
            twice = measure(fixtures()['large_unicode'] * 2)
        self.assertEqual(len(pages), 2 * pages_once)
        self.assertEqual(twice['recall_response_units'], once['recall_response_units'])
        self.assertEqual(once['recall_response_bytes'],
                         sum(len(observation_pack.wire(json.loads(p)).encode()) for p in pages[:pages_once]))
        self.assertGreater(once['recall_response_bytes'],
                           sum(len(observation_pack.wire(json.loads(p))) for p in pages[:pages_once]))

    def test_corrupted_recall_page_fails_roundtrip(self):
        real = observation_pack.recall
        def corrupt(*a):
            page = json.loads(real(*a))
            page['content'] += 'x'
            return json.dumps(page)
        with mock.patch.object(observation_pack, 'recall', corrupt):
            self.assertFalse(measure(fixtures()['large_unicode'])['recall_roundtrip'])

    def test_missing_tokenizer_fails_without_estimate_fallback(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(sys.modules, {'tiktoken': None}), \
                mock.patch.object(sys, 'argv', ['x', '--encoding', 'cl100k_base']), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('cannot load requested tokenizer', err.getvalue())
        self.assertEqual(out.getvalue(), '')

    def test_named_tokenizer_counts_and_label_reach_output(self):
        fake = mock.Mock()
        fake.get_encoding.return_value.encode.side_effect = lambda text, **kw: [0] * 7
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {'tiktoken': fake}), \
                mock.patch.object(sys, 'argv', ['x', '--encoding', 'cl100k_base', '--requests', '2']), \
                contextlib.redirect_stdout(out):
            main()
        fake.get_encoding.assert_called_once_with('cl100k_base')
        small = json.loads(out.getvalue())['small_control']
        self.assertEqual(small['counter'], 'tiktoken:cl100k_base:serialized_request_not_billing')
        self.assertEqual(small['control_units'], 14)

    def test_projection_mutating_source_transcript_is_rejected(self):
        real = observation_pack.project_request
        def mutate(source, root, sent):
            source['messages'][0]['content'] = 'tampered'
            return real(source, root, sent)
        with mock.patch.object(observation_pack, 'project_request', mutate):
            with self.assertRaisesRegex(AssertionError, 'mutated the source transcript'):
                measure(fixtures()['large_ascii'])

    def test_unknown_named_tokenizer_fails_without_estimate_fallback(self):
        fake = mock.Mock()
        fake.get_encoding.side_effect = ValueError('Unknown encoding nope')
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(sys.modules, {'tiktoken': fake}), \
                mock.patch.object(sys, 'argv', ['x', '--encoding', 'nope']), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('cannot load requested tokenizer', err.getvalue())
        self.assertEqual(out.getvalue(), '')

    def test_cli_rejects_non_positive_requests_before_measuring(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, 'argv', ['x', '--requests', '0']), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('--requests must be positive', err.getvalue())
        self.assertEqual(out.getvalue(), '')

    def test_one_corrupted_observation_fails_roundtrip_even_if_a_later_one_is_clean(self):
        real = observation_pack.recall
        bad = []
        def corrupt_first(root, observation_id, *rest):
            page = json.loads(real(root, observation_id, *rest))
            bad.append(observation_id) if not bad else None
            if observation_id == bad[0]:
                page['content'] += 'x'
            return json.dumps(page)
        rows = fixtures()['large_ascii'] + fixtures()['large_unicode']
        with mock.patch.object(observation_pack, 'recall', corrupt_first):
            result = measure(rows)
        self.assertEqual(result['recalled_observations'], 2)
        self.assertIs(result['recall_roundtrip'], False)

    def test_request_rows_are_numbered_from_one(self):
        rows = measure(fixtures()['large_ascii'], requests=3)['requests']
        self.assertEqual([row['request'] for row in rows], [1, 2, 3])

    def test_recall_error_and_stalled_pagination_are_rejected(self):
        real = observation_pack.recall
        def error(root, observation_id, *rest):
            return real(root, 'obs_' + '0' * 24, *rest)  # well-formed, absent
        calls = []
        def stalled(*a):
            calls.append(a)
            page = json.loads(real(*a))
            if len(calls) > 3:  # bounded: a missing guard fails instead of hanging
                return json.dumps(page)
            return json.dumps({**page, 'eof': False, 'next_offset': a[2]})
        for name, fake, message in (('error', error, 'Unknown observation id'), ('stalled', stalled, 'no progress')):
            with self.subTest(name=name), mock.patch.object(observation_pack, 'recall', fake):
                with self.assertRaisesRegex(AssertionError, message):
                    measure(fixtures()['large_ascii'])

    def test_named_tokenizer_does_not_reject_special_token_text(self):
        fake = mock.Mock()
        fake.get_encoding.return_value.encode.return_value = [0]
        with mock.patch.dict(sys.modules, {'tiktoken': fake}), \
                mock.patch.object(sys, 'argv', ['x', '--encoding', 'cl100k_base', '--requests', '1']), \
                contextlib.redirect_stdout(io.StringIO()):
            main()
        calls = fake.get_encoding.return_value.encode.call_args_list
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(call.kwargs, {'disallowed_special': ()})

    def test_default_cli_reports_every_fixture_with_estimate_counter(self):
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {'tiktoken': None}), \
                mock.patch.object(sys, 'argv', ['x', '--requests', '4']), \
                contextlib.redirect_stdout(out):
            main()
        report = json.loads(out.getvalue())
        self.assertEqual(set(report), set(fixtures()))
        for name, result in report.items():
            with self.subTest(name=name):
                self.assertEqual(result['counter'], 'ceil_unicode_chars_div_4_estimate')
                self.assertIsNone(result['provider_billed_tokens'])
                self.assertEqual(len(result['requests']), 4)
                self.assertTrue(result['recall_roundtrip'] and result['transcript_unchanged'])
                packed = name in ('large_ascii', 'large_unicode')
                self.assertEqual(result['saved_units_before_recall'] > 0, packed)
                self.assertEqual(result['recalled_observations'], int(packed))
