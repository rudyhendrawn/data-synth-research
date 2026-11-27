# CSV and Parquet datasets

CSV remains the source of truth and the default input format. Parquet is an optional verified copy: it can reduce storage and repeated parsing time, but does not change the model, split, feature set or oversampling ratio. Keep CSVs and conversion reports for reproducibility.

## Install and convert on the Mac

Install project dependencies, including `pyarrow`, in the project's virtual environment. The following command is an example; **it processes the actual datasets only when you run it**:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p /Users/rudyhendrawan/Projects/data/parquet
python scripts/convert_datasets_parquet.py \
  /Users/rudyhendrawan/Projects/data/01_creditcard.csv \
  /Users/rudyhendrawan/Projects/data/03_fraud_oracle.csv \
  /Users/rudyhendrawan/Projects/data/04_bank_account.csv \
  /Users/rudyhendrawan/Projects/data/05_online_payment.csv \
  --output-dir /Users/rudyhendrawan/Projects/data/parquet
```

The default compression is `snappy`; use `--compression zstd` or `--compression gzip` if desired. The original files are not modified. Existing output files or reports cause an error rather than being overwritten. For a preliminary measurement, supply only `05_online_payment.csv`; use a separate empty directory for any repeat conversion.

For every file the converter reads the complete CSV using the same pandas defaults as the normal loader, writes a temporary Parquet file, reads it back, and compares all values exactly, column order, row order, nulls, and pandas dtypes. Only after this check passes does it publish the Parquet copy and a `.parquet.report.json` report. The report contains SHA-256 hashes, row count, column names and dtypes, null counts, file sizes, read/write times, and deep DataFrame memory estimates.

The loader and verifier normalize Parquet `None` values to `NaN`, matching the CSV loader's representation of missing values before categorical encoding. Other Parquet consumers may represent nulls differently; use the project loader for comparable preprocessing.

Conversion currently holds both the CSV and restored Parquet DataFrames in memory. It may require more RAM than reading CSV alone. The recorded memory values are DataFrame sizes, **not peak process RAM**; the timings are single measurements and may be affected by disk caching and execution order. Use repeated measurements and OS-level peak memory monitoring to assess performance before extrapolating costs.

## Run revision experiments using Parquet

Transfer the four `.parquet` files to a directory on Runpod, for example `/workspace/data/parquet/`. The JSON configuration must retain its original dataset keys such as `05_online_payment.csv`; the loader maps those keys to corresponding `.parquet` filenames when explicitly selected:

```bash
python scripts/revision_experiments.py configs/revision_experiments.json \
  --data-root /workspace/data/parquet --data-format parquet \
  --output-dir /workspace/revision-results/full
```

Omit `--data-format parquet` to use the original CSV files. Missing Parquet files cause an error; there is no silent fallback to CSV. The selected format and data root are recorded in `environment.json`. Dataset identity in result tables remains the configured dataset name. Copy the conversion reports with the data or archive them alongside experiment results.

To use the loader in Python, set `UniversalDataLoader(..., data_format='parquet')`; the default is `csv`. The other ablation entry points currently use CSV by default and do not expose the format switch. Parquet chunk reading is supported by the loader's separate large-data iterator, but the revision benchmark currently reads complete datasets and does not use that iterator.

## Memory and leakage limits

Parquet compression primarily affects disk storage. Once decoded, preprocessing, train/validation/test splits, resampling and synthetic-data evaluation can allocate multiple in-memory copies. This change alone does not guarantee that Online Payment fits in 31 GB RAM. Keep the existing temporal split, label distribution, feature selection and oversampling ratios unchanged; profile later processing separately if necessary. Validate metrics and split assignments against a CSV pilot before comparing results across formats or versions.
