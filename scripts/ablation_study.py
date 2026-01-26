import logging
import os
import sys
from datetime import datetime

import pandas as pd

# Ensure project root is on sys.path for local imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from benchmark import AblationAnalyzer, AblationStudyManager, CrossDomainBenchmark
from preprocessor.data_config import DATASET_CONFIG


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ablation_study.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_datasets() -> dict[str, str]:
    datasets = {}
    for filename, cfg in DATASET_CONFIG.items():
        datasets[cfg["name"]] = filename
    return datasets


def log_experiments(label: str, experiments) -> None:
    logger.info("=== %s experiments ===", label)
    for exp in experiments:
        logger.info("exp_id=%s components=%s", exp.exp_id, exp.components)


if __name__ == "__main__":
    # data_root = os.environ.get("DATA_ROOT", os.path.join(project_root, "data"))
    data_root = "/Users/rudyhendrawan/Projects/data"

    datasets = build_datasets()

    ablation_mgr = AblationStudyManager(output_dir="results/ablation")
    benchmark = CrossDomainBenchmark(
        datasets=datasets,
        output_dir="results/cross_domain",
        random_seeds=[42, 123],
        data_root=data_root,
        precision_target=0.9,
        fpr_target=0.05,
        threshold_strategy="precision",
        bootstrap_samples=2000,
    )

    dataset_tag = "all_datasets"

    logger.info("Generating Ablation 1: Oversampling Methods")
    exp_oversampling = ablation_mgr.generate_single_factor_ablation("oversampling")
    ablation_mgr.save_experiments(
        exp_oversampling,
        "ablation_1_oversampling.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_1_oversampling",
    )
    log_experiments("Oversampling", exp_oversampling)

    logger.info("Generating Ablation 2: Model Architecture")
    exp_model = ablation_mgr.generate_single_factor_ablation("model")
    ablation_mgr.save_experiments(
        exp_model,
        "ablation_2_model.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_2_model",
    )
    log_experiments("Model", exp_model)

    logger.info("Generating Ablation 3: Oversampling × Model")
    exp_pairwise = ablation_mgr.generate_pairwise_ablation("oversampling", "model")
    ablation_mgr.save_experiments(
        exp_pairwise,
        "ablation_3_oversampling_model.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_3_oversampling_model",
    )
    log_experiments("Oversampling×Model", exp_pairwise)

    logger.info("Generating Ablation 4: Calibration Methods")
    exp_calibration = ablation_mgr.generate_single_factor_ablation("calibration")
    ablation_mgr.save_experiments(
        exp_calibration,
        "ablation_4_calibration.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_4_calibration",
    )
    log_experiments("Calibration", exp_calibration)

    logger.info("Generating Ablation 5: Anomaly Signals")
    exp_anomaly = ablation_mgr.generate_single_factor_ablation("anomaly_signal")
    ablation_mgr.save_experiments(
        exp_anomaly,
        "ablation_5_anomaly_signal.json",
        dataset_tag=dataset_tag,
        experiment_tag="ablation_5_anomaly_signal",
    )
    log_experiments("Anomaly", exp_anomaly)

    logger.info("\n" + "=" * 60)
    logger.info("Running Ablation Study: Oversampling Methods")
    logger.info("=" * 60)

    results_df = benchmark.run_ablation_study(
        exp_oversampling,
        project_root,
        experiment_tag="ablation_1_oversampling",
        dataset_tag=dataset_tag,
    )

    analyzer = AblationAnalyzer(results_df)
    baseline_exp = "ablation_oversampling_None"
    delta_df = analyzer.compute_delta_metrics(baseline_exp)

    print("\n" + "=" * 60)
    print("Delta Metrics (vs Baseline - No Oversampling)")
    print("=" * 60)
    print(delta_df.sort_values("delta_pr_auc_mean", ascending=False))

    analyzer.create_ablation_heatmap(
        component="oversampling",
        metric="pr_auc",
        save_path=f"results/ablation/heatmap_oversampling_prauc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )

    print("\n" + "=" * 60)
    print("Statistical Significance Tests (Oversampling vs None)")
    print("=" * 60)

    oversampling_baseline = "ablation_oversampling_None"
    significance_results = []
    exp_map = {exp.exp_id: exp.components for exp in exp_oversampling}

    for exp in exp_oversampling:
        if exp.exp_id == oversampling_baseline:
            continue
        for dataset in datasets.keys():
            test_result = analyzer.statistical_significance_test(
                exp_id_1=oversampling_baseline,
                exp_id_2=exp.exp_id,
                dataset=dataset,
                metric="pr_auc",
            )
            test_result["threshold_strategy"] = benchmark.threshold_strategy
            test_result["precision_target"] = benchmark.precision_target
            test_result["fpr_target"] = benchmark.fpr_target
            test_result["exp_1_components"] = exp_map.get(test_result["exp_1"], {})
            test_result["exp_2_components"] = exp_map.get(test_result["exp_2"], {})
            significance_results.append(test_result)

    if significance_results:
        sig_df = pd.DataFrame(significance_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sig_path = os.path.join(
            "results", "ablation", f"significance_oversampling_vs_none_{timestamp}.csv"
        )
        sig_df.to_csv(sig_path, index=False)
        logger.info("Oversampling significance tests saved to %s", sig_path)

    # === RUN MODEL ABLATION + SIGNIFICANCE (ALL MODELS VS BASELINE) ===
    logger.info("\n" + "=" * 60)
    logger.info("Running Ablation Study: Model Architecture")
    logger.info("=" * 60)

    model_results_df = benchmark.run_ablation_study(
        exp_model,
        project_root,
        experiment_tag="ablation_2_model",
        dataset_tag=dataset_tag,
    )
    model_analyzer = AblationAnalyzer(model_results_df)
    baseline_model_ids = [
        "ablation_model_LogisticRegression",
        "ablation_model_DecisionTree",
    ]

    model_sig_results = []
    model_exp_map = {exp.exp_id: exp.components for exp in exp_model}

    for exp in exp_model:
        if exp.exp_id in baseline_model_ids:
            continue
        for baseline_model_id in baseline_model_ids:
            for dataset in datasets.keys():
                test_result = model_analyzer.statistical_significance_test(
                    exp_id_1=baseline_model_id,
                    exp_id_2=exp.exp_id,
                    dataset=dataset,
                    metric="pr_auc",
                )
                test_result["threshold_strategy"] = benchmark.threshold_strategy
                test_result["precision_target"] = benchmark.precision_target
                test_result["fpr_target"] = benchmark.fpr_target
                test_result["exp_1_components"] = model_exp_map.get(test_result["exp_1"], {})
                test_result["exp_2_components"] = model_exp_map.get(test_result["exp_2"], {})
                model_sig_results.append(test_result)

    if model_sig_results:
        model_sig_df = pd.DataFrame(model_sig_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_sig_path = os.path.join(
            "results", "ablation", f"significance_models_vs_baseline_{timestamp}.csv"
        )
        model_sig_df.to_csv(model_sig_path, index=False)
        logger.info("Model significance tests saved to %s", model_sig_path)

    # === RUN PAIRWISE ABLATION (Oversampling × Model) ===
    logger.info("\n" + "=" * 60)
    logger.info("Running Ablation Study: Oversampling × Model (Pairwise)")
    logger.info("=" * 60)
    benchmark.run_ablation_study(
        exp_pairwise,
        project_root,
        experiment_tag="ablation_3_oversampling_model",
        dataset_tag=dataset_tag,
    )
