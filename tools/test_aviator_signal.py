"""Check past-only feature alignment and withholding unsupported alerts."""
import unittest
import numpy as np
from aviator_signal import HISTORY, auc, decide, features


class SignalTests(unittest.TestCase):
    def test_target_is_not_in_history(self):
        sequence = np.arange(1, 90, dtype=float)
        windows = np.lib.stride_tricks.sliding_window_view(sequence, HISTORY)[:-1]
        first = features(windows)[0]
        expected = features(sequence[:HISTORY])
        np.testing.assert_array_equal(first, expected)
        altered = sequence.copy()
        altered[HISTORY] = 999
        altered_windows = np.lib.stride_tricks.sliding_window_view(altered, HISTORY)[:-1]
        np.testing.assert_array_equal(features(altered_windows)[0], expected)

    def test_invalid_or_incomplete_history_rejected(self):
        for history in ([1.0]*63, [1.0]*63+[np.nan], [1.0]*63+[0.0]):
            with self.assertRaises(ValueError):
                features(history)

    def test_auc_handles_tied_predictions(self):
        self.assertEqual(auc(np.array([0, 1, 0, 1]), np.array([0.5]*4)), 0.5)
        self.assertEqual(auc(np.array([0, 1]), np.array([0.1, 0.9])), 1.0)

    def test_high_probability_does_not_bypass_validation(self):
        n = features([1.0]*HISTORY).shape[0]
        model = {'mean': [0.0]*n, 'scale': [1.0]*n, 'weights': [0.0]*n,
                 'intercept': 20.0, 'calibration_slope': 1.0,
                 'calibration_intercept': 0.0, 'threshold': 0.95,
                 'validation_gate_passed': False}
        result = decide(model, [1.0]*HISTORY)
        self.assertGreater(result['experimental_probability_below_2x'], 0.99)
        self.assertEqual(result['decision'], 'NO_SIGNAL')
        self.assertEqual(result['reason'], 'validation_gate_failed')
        model['validation_gate_passed'] = True
        result = decide(model, [1.0]*HISTORY)
        self.assertEqual(result['decision'], 'NO_SIGNAL')
        self.assertEqual(result['reason'], 'live_validation_pending')
        self.assertFalse(result['live_source_connected'])


if __name__ == '__main__':
    unittest.main()
