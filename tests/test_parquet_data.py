import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from loader.data_loader import UniversalDataLoader
from benchmark.cross_domain import CrossDomainBenchmark
from benchmark.temporal_cv import TemporalLeakageFreeCV
from preprocessor.data_config import DATASET_CONFIG


ROOT = Path(__file__).resolve().parents[1]


class ParquetDataTests(unittest.TestCase):
    def test_conversion_and_explicit_loader_preserve_rows_and_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / '03_fraud_oracle.csv'
            original = pd.DataFrame({'Age': [30.5, None, 25.0], 'Make': ['A', None, 'A'], 'FraudFound_P': [0, 1, 0]})
            original.to_csv(source, index=False)
            command = [sys.executable, str(ROOT / 'scripts/convert_datasets_parquet.py'), str(source), '--output-dir', str(root)]
            completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            loader = UniversalDataLoader('03_fraud_oracle.csv', str(ROOT), str(root), data_format='parquet')
            assert_frame_equal(pd.read_csv(source), loader.load_raw())
            chunks = list(loader._iter_raw_chunks(chunk_size=2))
            assert_frame_equal(pd.read_csv(source), pd.concat(chunks, ignore_index=True))
            report = json.loads((root / '03_fraud_oracle.parquet.report.json').read_text())
            self.assertEqual(report['rows'], 3)
            self.assertTrue(report['verified'])
            second = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(second.returncode, 0)

    def test_revision_runner_accepts_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'Age': list(range(100)), 'FraudFound_P': [0, 1] * 50}).to_parquet(root / '03_fraud_oracle.parquet', index=False)
            config = {'seeds': [42], 'precision_targets': [0.5],
                      'datasets': {'03_fraud_oracle.csv': [{'exp_id': 'pilot', 'components': {'model': 'LogisticRegression', 'oversampling': 'None'}}]},
                      'benchmark_options': {'bootstrap_samples': 2}}
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            completed = subprocess.run([sys.executable, str(ROOT / 'scripts/revision_experiments.py'),
                str(config_path), '--data-root', str(root), '--data-format', 'parquet', '--output-dir', str(root / 'output')],
                capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            runs = pd.read_csv(next((root / 'output').iterdir()) / 'runs.csv')
            self.assertEqual(runs.loc[0, 'status'], 'success')

    def test_default_loader_prefers_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'FraudFound_P': [0, 1]}).to_csv(root / '03_fraud_oracle.csv', index=False)
            loader = UniversalDataLoader('03_fraud_oracle.csv', str(ROOT), str(root))
            self.assertEqual(len(loader.load_raw()), 2)

    def test_temporal_split_is_identical_after_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({'Time': [number // 2 for number in range(200)], 'Class': [0, 1] * 100, 'Amount': range(200)})
            frame.to_csv(root / '01_creditcard.csv', index=False)
            frame.to_parquet(root / '01_creditcard.parquet', index=False)
            benchmark = CrossDomainBenchmark({'Credit Card Fraud': '01_creditcard.csv'}, output_dir=str(root))
            temporal = TemporalLeakageFreeCV(gap_config=benchmark.gap_configs['01_creditcard.csv'])
            config = DATASET_CONFIG['01_creditcard.csv']
            csv = UniversalDataLoader('01_creditcard.csv', str(ROOT), str(root)).load_raw()
            parquet = UniversalDataLoader('01_creditcard.csv', str(ROOT), str(root), data_format='parquet').load_raw()
            for csv_split, parquet_split in zip(benchmark._split_dataset(csv, config, temporal), benchmark._split_dataset(parquet, config, temporal)):
                assert_frame_equal(csv_split, parquet_split)


if __name__ == '__main__':
    unittest.main()
