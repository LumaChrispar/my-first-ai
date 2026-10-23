"""Verify temporal partitioning, calibration accounting, and holdout exclusion."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark_aviator import folds, improvement, load_development, reliability, selection_curve


class BenchmarkTests(unittest.TestCase):
    def test_windows_are_ordered_with_disjoint_evaluations(self):
        windows = folds(10000)
        previous_end = 0
        for fit_end, calibration_end, evaluation_end in windows:
            self.assertTrue(0 < fit_end < calibration_end < evaluation_end <= 10000)
            self.assertGreaterEqual(calibration_end, previous_end)
            previous_end = evaluation_end

    def test_loading_does_not_require_or_open_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'splits': {}}
            for name in ('train', 'validation'):
                raw = b'1.5\n3.0\n'
                (root / (name+'.txt')).write_bytes(raw)
                manifest['splits'][name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'rows': 2}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            values, _ = load_development(root)
            np.testing.assert_array_equal(values, [1.5, 3.0, 1.5, 3.0])
            (root / 'train.txt').write_bytes(b'99\n')
            with self.assertRaises(ValueError):
                load_development(root)

    def test_reliability_accounts_for_all_rows_and_endpoints(self):
        result = reliability(np.array([0., 0., 1., 1.]), np.array([0., .5, .5, 1.]))
        self.assertEqual(sum(b['rows'] for b in result['bins']), 4)
        self.assertEqual(result['ece_10_bins'], 0.)

    def test_coverage_and_precision_at_fixed_thresholds(self):
        curve = selection_curve(np.array([1., 0., 1.]), np.array([.96, .96, .50]))
        self.assertEqual(curve[-1]['count'], 2)
        self.assertEqual(curve[-1]['precision'], .5)
        self.assertAlmostEqual(curve[-1]['coverage'], 2/3)

    def test_baseline_comparison_has_correct_direction(self):
        y = np.tile([0., 1.], 100)
        baseline = np.full(len(y), .5)
        self.assertEqual(improvement(y, baseline, baseline)['brier_improvement'], 0.)
        self.assertEqual(improvement(y, y, baseline)['brier_improvement'], .25)


if __name__ == '__main__':
    unittest.main()
