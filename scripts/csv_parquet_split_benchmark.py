# This script only benchmarks CSV and Parquet splits using cross-domain and temporal leakage-free CV methods
# on dataset "05_online_payment.csv"

import sys
from pathlib import Path

from pandas.testing import assert_frame_equal

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from benchmark.cross_domain import CrossDomainBenchmark
from benchmark.temporal_cv import TemporalLeakageFreeCV
from loader.data_loader import UniversalDataLoader
from preprocessor.data_config import DATASET_CONFIG

dataset = "05_online_payment.csv"
config = DATASET_CONFIG[dataset]

benchmark = CrossDomainBenchmark(
    datasets={config["name"]: dataset},
    output_dir="/Users/rudyhendrawan/Projects/data/parquet-split-check"
)

temporal_cv = TemporalLeakageFreeCV(
    gap_config=benchmark.gap_configs[dataset]
)

csv_loader = UniversalDataLoader(
    dataset, str(project_root), "/Users/rudyhendrawan/Projects/data",
    verbose=True,
    data_format="csv"
)

parquet_loader = UniversalDataLoader(
    dataset, str(project_root), "/Users/rudyhendrawan/Projects/data/parquet",
    verbose=True,
    data_format="parquet"
)

csv_splits = benchmark._split_dataset(
    csv_loader.load_raw(), config, temporal_cv
)

parquet_splits = benchmark._split_dataset(
    parquet_loader.load_raw(), config, temporal_cv
)

for name, csv_part, parquet_part in zip (
    ("Train", "Validation", "Test"), csv_splits, parquet_splits
):
    assert_frame_equal(csv_part, parquet_part, check_dtype=True, check_exact=True)
    print(f"{name}: IDENTIK - {len(csv_part):,} baris")
    print(csv_part[config["label_col"]].value_counts().to_dict())

print("Semua split CSV dan Parquet identik.")
