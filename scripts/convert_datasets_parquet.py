import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def convert(source, output_dir, compression):
    if source.suffix.lower() != '.csv':
        raise ValueError(f'Expected a CSV file: {source}')
    output = output_dir / (source.stem + '.parquet')
    report_path = output.with_suffix('.parquet.report.json')
    if output.exists() or report_path.exists():
        raise FileExistsError(f'Output or report already exists: {output}')
    started = time.perf_counter()
    original = pd.read_csv(source)
    csv_read_seconds = time.perf_counter() - started
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix='.parquet', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        started = time.perf_counter()
        original.to_parquet(temporary_path, engine='pyarrow', compression=compression, index=False)
        write_seconds = time.perf_counter() - started
        started = time.perf_counter()
        restored = pd.read_parquet(temporary_path, engine='pyarrow').replace({None: np.nan})
        parquet_read_seconds = time.perf_counter() - started
        assert_frame_equal(original, restored, check_dtype=True, check_exact=True)
        report = {
            'source': str(source.resolve()), 'output': str(output.resolve()),
            'source_sha256': sha256_file(source), 'parquet_sha256': sha256_file(temporary_path),
            'rows': len(original), 'columns': list(original.columns),
            'dtypes': {column: str(dtype) for column, dtype in original.dtypes.items()},
            'null_counts': {column: int(value) for column, value in original.isna().sum().items()},
            'csv_bytes': source.stat().st_size, 'parquet_bytes': temporary_path.stat().st_size,
            'csv_read_seconds': csv_read_seconds, 'parquet_read_seconds': parquet_read_seconds,
            'parquet_write_seconds': write_seconds,
            'csv_dataframe_bytes': int(original.memory_usage(index=True, deep=True).sum()),
            'parquet_dataframe_bytes': int(restored.memory_usage(index=True, deep=True).sum()),
            'compression': compression, 'verified': True,
            'measurement_note': 'Single read per format in one process; DataFrame memory is not peak process memory.',
        }
        os.link(temporary_path, output)
        try:
            with report_path.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
        except Exception:
            output.unlink()
            raise
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Convert CSV datasets to verified Parquet files')
    parser.add_argument('csv_files', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--compression', choices=['snappy', 'zstd', 'gzip'], default='snappy')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.csv_files:
        convert(source, args.output_dir, args.compression)


if __name__ == '__main__':
    main()
