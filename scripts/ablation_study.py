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


if __name__ == "__main__":
    # data_root = os.environ.get("DATA_ROOT", os.path.join(project_root, "data"))
    data_root = "/Users/rudyhendrawan/Projects/data"

    datasets = build_datasets()

    ablation_mgr = AblationStudyManager(output_dir="results/ablation")
    benchmark = CrossDomainBenchmark(
        datasets=datasets,
        output_dir="results/cross_domain",
        random_seeds=[42, 123, 456],
        data_root=data_root,
        precision_target=0.9,
        fpr_target=0.05,
        threshold_strategy="precision",
        bootstrap_samples=2000,
    )

    logger.info("Generating Ablation 1: Oversampling Methods")
    exp_oversampling = ablation_mgr.generate_single_factor_ablation("oversampling")
    ablation_mgr.save_experiments(exp_oversampling, "ablation_1_oversampling.json")

    logger.info("Generating Ablation 2: Model Architecture")
    exp_model = ablation_mgr.generate_single_factor_ablation("model")
    ablation_mgr.save_experiments(exp_model, "ablation_2_model.json")

    logger.info("Generating Ablation 3: Oversampling × Model")
    exp_pairwise = ablation_mgr.generate_pairwise_ablation("oversampling", "model")
    ablation_mgr.save_experiments(exp_pairwise, "ablation_3_oversampling_model.json")

    logger.info("Generating Ablation 4: Calibration Methods")
    exp_calibration = ablation_mgr.generate_single_factor_ablation("calibration")
    ablation_mgr.save_experiments(exp_calibration, "ablation_4_calibration.json")

    logger.info("Generating Ablation 5: Anomaly Signals")
    exp_anomaly = ablation_mgr.generate_single_factor_ablation("anomaly_signal")
    ablation_mgr.save_experiments(exp_anomaly, "ablation_5_anomaly_signal.json")

    logger.info("\n" + "=" * 60)
    logger.info("Running Ablation Study: Oversampling Methods")
    logger.info("=" * 60)

    results_df = benchmark.run_ablation_study(exp_oversampling, project_root)

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
    print("Statistical Significance Tests")
    print("=" * 60)

    significance_results = []
    exp_map = {exp.exp_id: exp.components for exp in exp_oversampling}

    for dataset in datasets.keys():
        test_result = analyzer.statistical_significance_test(
            exp_id_1="ablation_oversampling_None",
            exp_id_2="ablation_oversampling_CTGAN",
            dataset=dataset,
            metric="pr_auc",
        )
        test_result["threshold_strategy"] = benchmark.threshold_strategy
        test_result["precision_target"] = benchmark.precision_target
        test_result["fpr_target"] = benchmark.fpr_target
        test_result["exp_1_components"] = exp_map.get(test_result["exp_1"], {})
        test_result["exp_2_components"] = exp_map.get(test_result["exp_2"], {})

        print(f"\n{dataset}:")
        print("  CTGAN vs Baseline")
        print(f"  Mean Δ PR-AUC: {test_result['mean_diff']:.4f}")
        print(f"  p-value: {test_result['p_value']:.4f}")
        print(f"  Significant: {test_result['is_significant']}")
        print(f"  Cohen's d: {test_result['effect_size_cohen_d']:.3f}")

        significance_results.append(test_result)

    if significance_results:
        sig_df = pd.DataFrame(significance_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sig_path = os.path.join("results", "ablation", f"significance_tests_{timestamp}.csv")
        sig_df.to_csv(sig_path, index=False)
        logger.info("Significance tests saved to %s", sig_path)
