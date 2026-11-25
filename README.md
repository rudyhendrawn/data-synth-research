# Data-Synth-Research

A reproducible research project for **fraud detection under class imbalance**, focused on:
- synthetic data generation (GAN-based oversampling),
- leakage-aware evaluation,
- cross-domain benchmarking,
- systematic ablation studies.

The project compares classical oversampling (ROS/SMOTE variants) and GAN-based methods (PyTorch GAN, CTGAN, Conditional WGAN-GP), with multiple classifiers and optional anomaly-score features.

## 1) Research Summary

This repository implements an end-to-end experimental framework to answer questions such as:
- When does synthetic oversampling improve fraud recall/PR-AUC?
- Which classifier families are most robust across datasets?
- How much gain comes from anomaly features (Isolation Forest / LOF)?
- Are improvements statistically significant vs baseline configurations?

Main principles used in the pipeline:
- **Leakage prevention** in splitting and preprocessing,
- **calibration + thresholding** for operational decision rules,
- **uncertainty estimation** using bootstrap confidence intervals,
- **experiment traceability** via JSON/CSV outputs and logs.

## 2) Dataset Sources

Datasets are not bundled in this repository. Download from the original sources:

- `01_creditcard.csv` (Credit Card Fraud):
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- `03_fraud_oracle.csv` (Vehicle Claim Fraud):
  https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
- `04_bank_account.csv` (Bank Account Fraud):
  https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- `05_online_payment.csv` (Online Payment Fraud):
  https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset

Place files under your data root (configured by `DATA_ROOT`).

## 3) Project Structure

```text
benchmark/              # Cross-domain benchmark engine, ablation generation, analysis
loader/                 # Dataset loading and split orchestration
preprocessor/           # Dataset configs + preprocessing logic
model/                  # GAN/CTGAN/WGAN oversamplers + anomaly models
evaluation/             # Metrics, calibration, synthetic-data evaluation
scripts/
  ablation_study.py     # Multi-dataset ablation runner
  single_dataset_ablation.py  # Full ablation runner for one selected dataset
results/                # Generated experiment outputs
notebooks/              # Analysis notebooks
```

## 4) Reproducible Setup

## Requirements

- Python 3.10+
- Recommended: virtual environment
- GPU optional (CUDA/MPS auto-detected by GAN modules; fallback to CPU)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment configuration

Create `.env` (and optional `.env.local` for machine-specific override).
Both main scripts load `.env`; if `.env.local` exists, it overrides values.

Example:

```env
DATA_ROOT="/path/to/datasets"
RUN_SINGLE_DATASET=True
RUN_FULL_ABLATION_SINGLE_DATASET=True
DATASET_NAME="03_fraud_oracle.csv"
ABLATION_SEEDS=42

USE_ANOMALY_FEATURE=True
ANOMALY_CONTAMINATION=0.01

GAN_TRAIN_MAX_MINORITY_RATIO=0.5
GAN_HIDDEN_DIM=64
GAN_NOISE_DIM=50
GAN_N_CRITIC=2
GAN_EPOCHS=100
GAN_BATCH_SIZE=128
GAN_CACHE_PATH="results/synthetic_cache"
GAN_EARLY_STOPPING=True
GAN_EARLY_STOPPING_PATIENCE=10
GAN_EARLY_STOPPING_DELTA=1e-3
```

## 5) Data Preparation Flow

High-level data preparation in this project:

1. **Read raw CSV** with dataset-specific schema from `preprocessor/data_config.py`.
2. **Select split strategy**:
   - Temporal split with gap (purging/embargo) when temporal column is configured and `shuffle=False`.
   - Stratified split when no temporal strategy is applicable.
   - Group-aware split utility is available in `loader/data_loader.py` when a valid group column is configured.
3. **Create train/validation/test partitions**.
4. **Do not fit transformations on validation/test** (fit only on train, transform on val/test).

For benchmark runs (`CrossDomainBenchmark`), temporal leakage-free split is handled by `TemporalLeakageFreeCV` with dataset-specific `TimeGapConfig`.

### Add a New Dataset Config (`preprocessor/data_config.py`)

To onboard a new dataset, add one new entry into `DATASET_CONFIG` using the CSV filename as the key.

1. Put your CSV file under `DATA_ROOT`.
2. Add a new config block in `preprocessor/data_config.py`.
3. Ensure `label_col` exists in the CSV.
4. Choose splitting behavior:
   - Temporal split: set `temporal_col` and `shuffle=False`
   - Stratified random split: set `shuffle=True` and `stratify=True`
   - Group-aware split (optional in loader flow): set `group_col` and `group_min_interactions`
5. List categorical columns in `categorical_cols` (or keep empty for fully numeric data).
6. Set `drop_cols` for IDs/PII/leaky columns you do not want as features.
7. Run one dry test (`python3 scripts/single_dataset_ablation.py`) and verify logs for split + preprocessing.

Template:

```python
DATASET_CONFIG = {
    # ...existing datasets...
    "06_new_dataset.csv": {
        "name": "New Fraud Dataset",
        "type": "mixed",  # e.g., "numeric_temporal" or "mixed"
        "label_col": "isFraud",
        "drop_cols": ["customer_id"],  # optional
        "temporal_col": "event_time",   # set None if not using temporal split
        "requires_encoding": True,
        "categorical_cols": ["channel", "merchant_type"],
        "group_col": None,              # optional: e.g., "account_id"
        "group_min_interactions": 2,    # optional, used if group_col is set
        "test_size": 0.4,
        "stratify": True,
        "shuffle": False,               # False for temporal split, True for random split
    },
}
```

Practical notes:
- If `temporal_col` is not available, use `temporal_col=None` and `shuffle=True`.
- If unseen categories appear in validation/test, the preprocessor maps them to `__UNK__`.
- Keep file key and `.env` value consistent, e.g. `DATASET_NAME="06_new_dataset.csv"`.

## 6) Data Preprocessing Flow

Preprocessing is handled by `DatasetPreprocessor`:

1. Drop configured columns (`drop_cols`).
2. Detect numeric/categorical features (or use explicit categorical config).
3. Encode categorical columns with `LabelEncoder`.
   - Unknown categories in val/test are mapped to `__UNK__` token.
4. Scale numeric columns using `StandardScaler`.
   - Scaler is **fit on training data only**.

This design prevents leakage from validation/test into preprocessing statistics.

## 7) Experiment Flow

Each experiment is a combination of components:

- **Oversampling**: `None`, `ROS`, `SMOTE`, `BorderlineSMOTE`, `SMOTEENN`, `PytorchGAN`, `CTGAN`, `ConditionalWGAN-GP`
- **Anomaly signal**: `None`, `IsolationForest`, `LOF`
- **Classifier**: `LogisticRegression`, `DecisionTree`, `RandomForest`, `XGBoost`, `MLP`
- **Calibration**: `None`, `Platt`, `Isotonic`

Execution sequence per experiment:

1. Load + split data (leakage-aware).
2. Fit preprocessor on train, transform val/test.
3. Optionally append anomaly score feature.
4. Apply selected oversampling method.
5. Train selected classifier.
6. Calibrate probabilities on validation set.
7. Select threshold using strategy:
   - Precision-target (`Recall@Precision>=P0`) or
   - FPR-target (`Precision@FPR<=alpha` style control).
8. Evaluate on test set.
9. Store metrics + metadata.

## Main evaluation metrics

- PR-AUC (primary)
- Precision, Recall, F1
- Recall@Precision target
- Lift@Top-1%
- Calibration/threshold metrics at selected threshold
- PR-AUC bootstrap CI (`bootstrap_pr_auc`)
- Synthetic-quality metrics (when synthetic samples exist):
  - KS mean,
  - correlation gap,
  - duplicate rate,
  - TSTR metrics.

## Statistical significance

`AblationAnalyzer.statistical_significance_test` runs paired t-tests between experiment pairs and stores:
- p-value,
- significance flag,
- effect size (Cohen's d),
- component metadata.

## 8) How to Run

### Reviewer revision experiments

Use `configs/revision_experiments.example.json` as a starting point for a separate configuration. Specify the configurations selected from the original results for each dataset before running; the example is illustrative and does not identify all reported winners. Set `baseline_ids` for paired PR-AUC differences and list the desired seeds and validation precision targets. Additional `CrossDomainBenchmark` options can be specified under `benchmark_options`.

```bash
python3 scripts/revision_experiments.py configs/revision_experiments.example.json --data-root /path/to/datasets
```

Each run writes its configuration, environment versions, per-run status and elapsed time, sensitivity results, and successful-run summaries under a timestamped directory in `results/revision/`. GAN caches are isolated by run, dataset, experiment and seed. Precision thresholds are chosen on validation data and evaluated on test data; inspect `validation_target_met` and `test_target_met` before interpreting a target. Pairwise differences use matching seeds; missing runs reduce `n_pairs`. Bootstrap intervals represent test-sample uncertainty, while variation between seeds reflects training randomness. Fixed seeds alone do not guarantee identical GPU results across hardware or software versions.

## A) Full cross-domain ablation (all datasets)

```bash
python3 scripts/ablation_study.py
```

## B) Full ablation on one selected dataset

```bash
python3 scripts/single_dataset_ablation.py
```

Use `.env/.env.local` to switch dataset and runtime knobs.

## 9) Output Artifacts

Generated files are saved under:

- `results/cross_domain/`
  - experiment result CSVs, e.g. `<dataset_tag>_<experiment_tag>_<timestamp>.csv`
- `results/ablation/`
  - ablation configuration JSONs,
  - significance test CSVs.
- project root logs:
  - `ablation_study.log`
  - `single_dataset_ablation.log`
  - `computation_time.log`

## 10) Reproducibility Checklist

1. Use fixed `ABLATION_SEEDS`.
2. Keep dataset filenames exactly as configured in `preprocessor/data_config.py`.
3. Save and version `.env` template used for each run.
4. Archive generated JSON + CSV + log files for each experiment batch.
5. For fair hardware comparisons, record whether runs used CPU/CUDA/MPS.

## License

See `LICENSE` for repository license terms. Dataset licenses follow each dataset source.
