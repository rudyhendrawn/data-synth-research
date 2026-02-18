import atexit
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

# Ensure project root is on sys.path for local imports.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from benchmark import AblationAnalyzer, AblationStudyManager, CrossDomainBenchmark
from preprocessor.data_config import DATASET_CONFIG


def _load_env_file(path: str, overwrite: bool = False) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if overwrite or key not in os.environ:
                os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw.strip())


_load_env_file(os.path.join(project_root, ".env"))
_load_env_file(os.path.join(project_root, ".env.local"), overwrite=True)

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(project_root, "data"))
DATASET_NAME = os.environ.get("DATASET_NAME", "03_fraud_oracle.csv")
RUN_FULL_ABLATION_SINGLE_DATASET = _env_bool("RUN_FULL_ABLATION_SINGLE_DATASET", True)
RUN_SIGNIFICANCE_TESTS = _env_bool("RUN_SIGNIFICANCE_TESTS", True)
ABLATION_SEED = _env_int(
    "ABLATION_SEED",
    _env_int("ABLATION_SEEDS", 42),  # Backward-compatible with older key
)
USE_ANOMALY_FEATURE = _env_bool("USE_ANOMALY_FEATURE", True)
ANOMALY_CONTAMINATION = _env_float("ANOMALY_CONTAMINATION", 0.01)

GAN_TRAIN_MAX_MINORITY_RATIO = _env_float("GAN_TRAIN_MAX_MINORITY_RATIO", 0.5)
GAN_HIDDEN_DIM = _env_int("GAN_HIDDEN_DIM", 64)
GAN_NOISE_DIM = _env_int("GAN_NOISE_DIM", 50)
GAN_N_CRITIC = _env_int("GAN_N_CRITIC", 2)
GAN_EPOCHS = _env_int("GAN_EPOCHS", 100)
GAN_BATCH_SIZE = _env_int("GAN_BATCH_SIZE", 128)
GAN_CACHE_PATH = os.environ.get(
    "GAN_CACHE_PATH",
    os.path.join(project_root, "results", "synthetic_cache"),
)
GAN_EARLY_STOPPING = _env_bool("GAN_EARLY_STOPPING", True)
GAN_EARLY_STOPPING_PATIENCE = _env_int("GAN_EARLY_STOPPING_PATIENCE", 10)
GAN_EARLY_STOPPING_DELTA = _env_float("GAN_EARLY_STOPPING_DELTA", 1e-3)

SCRIPT_START_TIME = time.perf_counter()


def _append_computation_time(script_name: str) -> None:
    elapsed_seconds = time.perf_counter() - SCRIPT_START_TIME
    log_path = os.path.join(project_root, "computation_time.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"{timestamp} | script={script_name} | elapsed_seconds={elapsed_seconds:.3f}\n"
        )


atexit.register(_append_computation_time, "scripts/single_dataset_ablation.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(project_root, "single_dataset_ablation.log")),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def _save_significance_results(
    analyzer: AblationAnalyzer,
    experiments,
    datasets: dict[str, str],
    benchmark: CrossDomainBenchmark,
    baseline_ids: list[str],
    filename_prefix: str,
) -> None:
    results = []
    exp_map = {exp.exp_id: exp.components for exp in experiments}
    baselines = set(baseline_ids)

    for exp in experiments:
        for baseline_id in baseline_ids:
            if exp.exp_id == baseline_id:
                continue
            if exp.exp_id in baselines and len(baseline_ids) > 1:
                continue
            for dataset in datasets.keys():
                row = analyzer.statistical_significance_test(
                    exp_id_1=baseline_id,
                    exp_id_2=exp.exp_id,
                    dataset=dataset,
                    metric="pr_auc",
                )
                row["threshold_strategy"] = benchmark.threshold_strategy
                row["precision_target"] = benchmark.precision_target
                row["fpr_target"] = benchmark.fpr_target
                row["exp_1_components"] = exp_map.get(row["exp_1"], {})
                row["exp_2_components"] = exp_map.get(row["exp_2"], {})
                results.append(row)

    if results:
        out_dir = os.path.join(project_root, "results", "ablation")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"{filename_prefix}_{timestamp}.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("Significance tests saved to %s", out_path)


if __name__ == "__main__":
    if DATASET_NAME not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown DATASET_NAME='{DATASET_NAME}'. Available: {list(DATASET_CONFIG.keys())}"
        )

    if not RUN_FULL_ABLATION_SINGLE_DATASET:
        logger.warning(
            "RUN_FULL_ABLATION_SINGLE_DATASET=0 detected. "
            "This script is intended for full single-dataset ablation; exiting."
        )
        raise SystemExit(0)

    dataset_label = DATASET_CONFIG[DATASET_NAME]["name"]
    datasets = {dataset_label: DATASET_NAME}
    dataset_tag = DATASET_NAME.replace(".csv", "")

    logger.info("Run mode: single-dataset full ablation")
    logger.info("Dataset: %s (%s)", dataset_label, DATASET_NAME)
    logger.info(
        (
            "Config: seed=%s | use_anomaly=%s | anomaly_contamination=%s "
            "| gan_epochs=%s | gan_batch=%s | gan_hidden=%s | gan_noise=%s | gan_n_critic=%s"
        ),
        ABLATION_SEED,
        USE_ANOMALY_FEATURE,
        ANOMALY_CONTAMINATION,
        GAN_EPOCHS,
        GAN_BATCH_SIZE,
        GAN_HIDDEN_DIM,
        GAN_NOISE_DIM,
        GAN_N_CRITIC,
    )

    ablation_mgr = AblationStudyManager(
        output_dir=os.path.join(project_root, "results", "ablation")
    )
    benchmark = CrossDomainBenchmark(
        datasets=datasets,
        output_dir=os.path.join(project_root, "results", "cross_domain"),
        random_seeds=[ABLATION_SEED],
        data_root=DATA_ROOT,
        use_anomaly_feature=USE_ANOMALY_FEATURE,
        anomaly_contamination=ANOMALY_CONTAMINATION,
        precision_target=0.9,
        fpr_target=0.05,
        threshold_strategy="precision",
        bootstrap_samples=2000,
        gan_train_max_minority_ratio=GAN_TRAIN_MAX_MINORITY_RATIO,
        gan_hidden_dim=GAN_HIDDEN_DIM,
        gan_noise_dim=GAN_NOISE_DIM,
        gan_n_critic=GAN_N_CRITIC,
        gan_epochs=GAN_EPOCHS,
        gan_batch_size=GAN_BATCH_SIZE,
        gan_cache_path=GAN_CACHE_PATH,
        gan_early_stopping=GAN_EARLY_STOPPING,
        gan_early_stopping_patience=GAN_EARLY_STOPPING_PATIENCE,
        gan_early_stopping_delta=GAN_EARLY_STOPPING_DELTA,
    )

    exp_oversampling = ablation_mgr.generate_single_factor_ablation("oversampling")
    exp_model = ablation_mgr.generate_single_factor_ablation("model")
    exp_anomaly = ablation_mgr.generate_single_factor_ablation("anomaly_signal")
    exp_calibration = ablation_mgr.generate_single_factor_ablation("calibration")
    exp_pairwise = ablation_mgr.generate_pairwise_ablation("oversampling", "model")

    ablation_mgr.save_experiments(
        exp_oversampling,
        "ablation_oversampling_single_dataset.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_oversampling_single_dataset",
    )
    ablation_mgr.save_experiments(
        exp_model,
        "ablation_model_single_dataset.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_model_single_dataset",
    )
    ablation_mgr.save_experiments(
        exp_anomaly,
        "ablation_anomaly_single_dataset.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_anomaly_single_dataset",
    )
    ablation_mgr.save_experiments(
        exp_calibration,
        "ablation_calibration_single_dataset.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_calibration_single_dataset",
    )
    ablation_mgr.save_experiments(
        exp_pairwise,
        "ablation_pairwise_single_dataset.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_pairwise_single_dataset",
    )

    oversampling_df = benchmark.run_ablation_study(
        exp_oversampling,
        project_root,
        experiment_tag="ablation_oversampling_single_dataset",
        dataset_tag=dataset_tag,
    )
    model_df = benchmark.run_ablation_study(
        exp_model,
        project_root,
        experiment_tag="ablation_model_single_dataset",
        dataset_tag=dataset_tag,
    )
    anomaly_df = benchmark.run_ablation_study(
        exp_anomaly,
        project_root,
        experiment_tag="ablation_anomaly_single_dataset",
        dataset_tag=dataset_tag,
    )
    calibration_df = benchmark.run_ablation_study(
        exp_calibration,
        project_root,
        experiment_tag="ablation_calibration_single_dataset",
        dataset_tag=dataset_tag,
    )
    benchmark.run_ablation_study(
        exp_pairwise,
        project_root,
        experiment_tag="ablation_pairwise_single_dataset",
        dataset_tag=dataset_tag,
    )

    if RUN_SIGNIFICANCE_TESTS:
        _save_significance_results(
            analyzer=AblationAnalyzer(oversampling_df),
            experiments=exp_oversampling,
            datasets=datasets,
            benchmark=benchmark,
            baseline_ids=["ablation_oversampling_None"],
            filename_prefix="significance_oversampling_single",
        )
        _save_significance_results(
            analyzer=AblationAnalyzer(model_df),
            experiments=exp_model,
            datasets=datasets,
            benchmark=benchmark,
            baseline_ids=["ablation_model_LogisticRegression", "ablation_model_DecisionTree"],
            filename_prefix="significance_models_single",
        )
        _save_significance_results(
            analyzer=AblationAnalyzer(anomaly_df),
            experiments=exp_anomaly,
            datasets=datasets,
            benchmark=benchmark,
            baseline_ids=["ablation_anomaly_signal_None"],
            filename_prefix="significance_anomaly_single",
        )
        _save_significance_results(
            analyzer=AblationAnalyzer(calibration_df),
            experiments=exp_calibration,
            datasets=datasets,
            benchmark=benchmark,
            baseline_ids=["ablation_calibration_None"],
            filename_prefix="significance_calibration_single",
        )
