import copy
import unittest

from benchmarks.observation_pack import fixtures, measure


class MeasurementTests(unittest.TestCase):
    def test_large_rows_preserve_first_two_and_recall_exactly(self):
        for name in ('large_ascii', 'large_unicode'):
            with self.subTest(name=name):
                source = fixtures()[name]
                original = copy.deepcopy(source)
                result = measure(source)
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

    def test_recall_can_outweigh_gross_savings(self):
        result = measure(fixtures()['large_ascii'], requests=3)
        self.assertLess(result['net_saved_units_after_one_full_recall_response'], 0)

    def test_invalid_request_count(self):
        with self.assertRaises(ValueError):
            measure([], requests=0)
