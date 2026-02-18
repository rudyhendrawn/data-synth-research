import os
import sys
import logging
import time
import atexit
import pandas as pd
from datetime import datetime

# Ensure project root is on sys.path for local imports
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

# Global configuration (from .env when present)
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(project_root, "data"))
RUN_SINGLE_DATASET = _env_bool("RUN_SINGLE_DATASET", False)
RUN_FULL_ABLATION_SINGLE_DATASET = _env_bool(
    "RUN_FULL_ABLATION_SINGLE_DATASET",
    RUN_SINGLE_DATASET,
)
DATASET_NAME = os.environ.get("DATASET_NAME", "03_fraud_oracle.csv")
ABLATION_SEED = _env_int("ABLATION_SEEDS", 42)
USE_ANOMALY_FEATURE = _env_bool("USE_ANOMALY_FEATURE", True)
ANOMALY_CONTAMINATION = _env_float(
    "ANOMALY_CONTAMINATION",
    0.01,
)
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


atexit.register(_append_computation_time, "scripts/ablation_study.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(project_root, "ablation_study.log")),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def build_datasets() -> dict[str, str]:
    datasets = {}
    for filename, cfg in DATASET_CONFIG.items():
        datasets[cfg["name"]] = filename
    return datasets


def build_selected_datasets(
    run_single_dataset: bool,
    dataset_name: str,
) -> tuple[dict[str, str], str]:
    if run_single_dataset:
        cfg = DATASET_CONFIG[dataset_name]
        return {cfg["name"]: dataset_name}, dataset_name.replace(".csv", "")
    return build_datasets(), "all_datasets"


def log_experiments(label: str, experiments) -> None:
    logger.info("=== %s experiments ===", label)
    for exp in experiments:
        logger.info("exp_id=%s components=%s", exp.exp_id, exp.components)


def save_significance_results(
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
                test_result = analyzer.statistical_significance_test(
                    exp_id_1=baseline_id,
                    exp_id_2=exp.exp_id,
                    dataset=dataset,
                    metric="pr_auc",
                )
                test_result["threshold_strategy"] = benchmark.threshold_strategy
                test_result["precision_target"] = benchmark.precision_target
                test_result["fpr_target"] = benchmark.fpr_target
                test_result["exp_1_components"] = exp_map.get(test_result["exp_1"], {})
                test_result["exp_2_components"] = exp_map.get(test_result["exp_2"], {})
                results.append(test_result)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(project_root, "results", "ablation")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{filename_prefix}_{timestamp}.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info("Significance tests saved to %s", out_path)


if __name__ == "__main__":
    run_single_dataset = RUN_SINGLE_DATASET or RUN_FULL_ABLATION_SINGLE_DATASET
    datasets, dataset_tag = build_selected_datasets(run_single_dataset, DATASET_NAME)
    scope_suffix = "single" if run_single_dataset else "all"
    logger.info("Run mode: %s", "single-dataset" if run_single_dataset else "all-datasets")
    logger.info("Datasets: %s", datasets)
    logger.info(
        (
            "Config: seeds=%s | use_anomaly=%s | anomaly_contamination=%s "
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

    logger.info("Generating Ablation: Oversampling Methods")
    exp_oversampling = ablation_mgr.generate_single_factor_ablation("oversampling")
    ablation_mgr.save_experiments(
        exp_oversampling,
        "ablation_oversampling.json",
        dataset_tag=dataset_tag,
        experiment_tag=f"ablation_oversampling_{scope_suffix}_dataset",
    )
    log_experiments("Oversampling", exp_oversampling)

    logger.info("Generating Ablation: Model Architecture")
    exp_model = ablation_mgr.generate_single_factor_ablation("model")
    ablation_mgr.save_experiments(
        exp_model,
        "ablation_model.json",
        dataset_tag=dataset_tag,
        experiment_tag=f"ablation_model_{scope_suffix}_dataset",
    )
    log_experiments("Model", exp_model)

    logger.info("Generating Ablation: Anomaly Signals")
    exp_anomaly = ablation_mgr.generate_single_factor_ablation("anomaly_signal")
    ablation_mgr.save_experiments(
        exp_anomaly,
        "ablation_anomaly_signal.json",
        dataset_tag=dataset_tag,
        experiment_tag=f"ablation_anomaly_{scope_suffix}_dataset",
    )
    log_experiments("Anomaly", exp_anomaly)

    logger.info("Generating Ablation: Calibration Methods")
    exp_calibration = ablation_mgr.generate_single_factor_ablation("calibration")
    ablation_mgr.save_experiments(
        exp_calibration,
        "ablation_calibration.json",
        dataset_tag=dataset_tag,
        experiment_tag=f"ablation_calibration_{scope_suffix}_dataset",
    )
    log_experiments("Calibration", exp_calibration)

    logger.info("Generating Ablation: Oversampling × Model")
    exp_pairwise = ablation_mgr.generate_pairwise_ablation("oversampling", "model")
    ablation_mgr.save_experiments(
        exp_pairwise,
        "ablation_pairwise_oversampling_model.json",
        dataset_tag=dataset_tag,
        experiment_tag=f"ablation_pairwise_{scope_suffix}_dataset",
    )
    log_experiments("Oversampling×Model", exp_pairwise)

    logger.info("%s", "=" * 60)
    logger.info("Running Ablation Study: Oversampling")
    logger.info("%s", "=" * 60)
    oversampling_df = benchmark.run_ablation_study(
        exp_oversampling,
        project_root,
        experiment_tag=f"ablation_oversampling_{scope_suffix}_dataset",
        dataset_tag=dataset_tag,
    )
    save_significance_results(
        analyzer=AblationAnalyzer(oversampling_df),
        experiments=exp_oversampling,
        datasets=datasets,
        benchmark=benchmark,
        baseline_ids=["ablation_oversampling_None"],
        filename_prefix=f"significance_oversampling_{scope_suffix}",
    )

    logger.info("%s", "=" * 60)
    logger.info("Running Ablation Study: Model")
    logger.info("%s", "=" * 60)
    model_df = benchmark.run_ablation_study(
        exp_model,
        project_root,
        experiment_tag=f"ablation_model_{scope_suffix}_dataset",
        dataset_tag=dataset_tag,
    )
    save_significance_results(
        analyzer=AblationAnalyzer(model_df),
        experiments=exp_model,
        datasets=datasets,
        benchmark=benchmark,
        baseline_ids=["ablation_model_LogisticRegression", "ablation_model_DecisionTree"],
        filename_prefix=f"significance_models_{scope_suffix}",
    )

    logger.info("%s", "=" * 60)
    logger.info("Running Ablation Study: Anomaly")
    logger.info("%s", "=" * 60)
    anomaly_df = benchmark.run_ablation_study(
        exp_anomaly,
        project_root,
        experiment_tag=f"ablation_anomaly_{scope_suffix}_dataset",
        dataset_tag=dataset_tag,
    )
    save_significance_results(
        analyzer=AblationAnalyzer(anomaly_df),
        experiments=exp_anomaly,
        datasets=datasets,
        benchmark=benchmark,
        baseline_ids=["ablation_anomaly_signal_None"],
        filename_prefix=f"significance_anomaly_{scope_suffix}",
    )

    logger.info("%s", "=" * 60)
    logger.info("Running Ablation Study: Calibration")
    logger.info("%s", "=" * 60)
    calibration_df = benchmark.run_ablation_study(
        exp_calibration,
        project_root,
        experiment_tag=f"ablation_calibration_{scope_suffix}_dataset",
        dataset_tag=dataset_tag,
    )
    save_significance_results(
        analyzer=AblationAnalyzer(calibration_df),
        experiments=exp_calibration,
        datasets=datasets,
        benchmark=benchmark,
        baseline_ids=["ablation_calibration_None"],
        filename_prefix=f"significance_calibration_{scope_suffix}",
    )

    logger.info("%s", "=" * 60)
    logger.info("Running Ablation Study: Oversampling × Model (Pairwise)")
    logger.info("%s", "=" * 60)
    benchmark.run_ablation_study(
        exp_pairwise,
        project_root,
        experiment_tag=f"ablation_pairwise_{scope_suffix}_dataset",
        dataset_tag=dataset_tag,
    )
