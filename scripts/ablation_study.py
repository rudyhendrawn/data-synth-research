import logging
import os

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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.environ.get("DATA_ROOT", os.path.join(project_root, "data"))

    datasets = build_datasets()

    ablation_mgr = AblationStudyManager(output_dir="results/ablation")
    benchmark = CrossDomainBenchmark(
        datasets=datasets,
        output_dir="results/cross_domain",
        random_seeds=[42, 123, 456, 789, 1011],
        data_root=data_root,
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
        save_path="results/ablation/heatmap_oversampling_prauc.png",
    )

    print("\n" + "=" * 60)
    print("Statistical Significance Tests")
    print("=" * 60)

    for dataset in datasets.keys():
        test_result = analyzer.statistical_significance_test(
            exp_id_1="ablation_oversampling_None",
            exp_id_2="ablation_oversampling_CTGAN",
            dataset=dataset,
            metric="pr_auc",
        )

        print(f"\n{dataset}:")
        print("  CTGAN vs Baseline")
        print(f"  Mean Δ PR-AUC: {test_result['mean_diff']:.4f}")
        print(f"  p-value: {test_result['p_value']:.4f}")
        print(f"  Significant: {test_result['is_significant']}")
        print(f"  Cohen's d: {test_result['effect_size_cohen_d']:.3f}")
