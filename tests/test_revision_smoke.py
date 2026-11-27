import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import metrics_at_threshold, select_threshold_by_precision


ROOT = Path(__file__).resolve().parents[1]


class PrecisionThresholdTests(unittest.TestCase):
    def test_threshold_metrics_match_predictions(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.05, 0.25, 0.4, 0.6, 0.8, 0.9])
        for target in (0.5, 0.9, 1.0):
            threshold, precision, recall = select_threshold_by_precision(labels, probabilities, target)
            measured = metrics_at_threshold(labels, probabilities, threshold)
            self.assertAlmostEqual(precision, measured['precision'])
            self.assertAlmostEqual(recall, measured['recall'])

    def test_unattainable_target_has_lower_precision(self):
        labels = np.array([1, 0, 1, 0])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        _, precision, _ = select_threshold_by_precision(labels, probabilities, 1.0)
        self.assertLess(precision, 1.0)


class RevisionRunnerTests(unittest.TestCase):
    def test_two_seed_synthetic_pilot_writes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rng = np.random.default_rng(42)
            pd.DataFrame({'Age': rng.normal(size=200), 'FraudFound_P': [0, 1] * 100}).to_csv(
                path / '03_fraud_oracle.csv', index=False
            )
            config = {
                'seeds': [42, 123], 'precision_targets': [0.5, 0.9],
                'datasets': {'03_fraud_oracle.csv': [
                    {'exp_id': 'pilot', 'components': {'model': 'LogisticRegression', 'oversampling': 'None'}}
                ]},
                'benchmark_options': {'bootstrap_samples': 4},
            }
            config_path = path / 'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            output = path / 'output'
            completed = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/revision_experiments.py'), str(config_path),
                 '--data-root', str(path), '--output-dir', str(output)],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run = next(output.iterdir())
            results = pd.read_csv(run / 'runs.csv')
            sensitivity = pd.read_csv(run / 'sensitivity_03_fraud_oracle.csv')
            self.assertEqual(list(results['status']), ['success', 'success'])
            self.assertEqual(set(results['seed']), {42, 123})
            self.assertEqual(len(sensitivity), 4)
            self.assertTrue((run / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
