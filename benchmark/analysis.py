from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class AblationAnalyzer:
    """Analyze and visualize ablation study results."""

    def __init__(self, results_df: pd.DataFrame) -> None:
        self.results_df = results_df

    def compute_delta_metrics(self, baseline_exp_id: str) -> pd.DataFrame:
        """
        Compute Δ metrics relative to baseline.
        """
        baseline = self.results_df[self.results_df["exp_id"] == baseline_exp_id].set_index(
            ["dataset", "seed"]
        )

        delta_results = []
        for exp_id in self.results_df["exp_id"].unique():
            if exp_id == baseline_exp_id:
                continue

            exp_data = self.results_df[self.results_df["exp_id"] == exp_id].set_index(
                ["dataset", "seed"]
            )

            delta_pr_auc = exp_data["pr_auc"] - baseline["pr_auc"]
            delta_recall = exp_data["recall"] - baseline["recall"]
            delta_f1 = exp_data["f1"] - baseline["f1"]

            for dataset in delta_pr_auc.index.get_level_values("dataset").unique():
                delta_results.append(
                    {
                        "exp_id": exp_id,
                        "dataset": dataset,
                        "delta_pr_auc_mean": delta_pr_auc.loc[dataset].mean(),
                        "delta_pr_auc_std": delta_pr_auc.loc[dataset].std(),
                        "delta_recall_mean": delta_recall.loc[dataset].mean(),
                        "delta_recall_std": delta_recall.loc[dataset].std(),
                        "delta_f1_mean": delta_f1.loc[dataset].mean(),
                        "delta_f1_std": delta_f1.loc[dataset].std(),
                    }
                )

        return pd.DataFrame(delta_results)

    def create_ablation_heatmap(
        self,
        component: str,
        metric: str = "pr_auc",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create heatmap showing effect of ablating a component across datasets.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        pivot_data = self.results_df.groupby([component, "dataset"])[metric].mean().unstack()

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=pivot_data.mean().mean(),
            cbar_kws={"label": metric.upper()},
        )
        plt.title(f"Ablation Study: {component.capitalize()} × Dataset ({metric.upper()})")
        plt.xlabel("Dataset")
        plt.ylabel(component.capitalize())
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def statistical_significance_test(
        self,
        exp_id_1: str,
        exp_id_2: str,
        dataset: str,
        metric: str = "pr_auc",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Paired t-test between two experiments on a dataset.
        """
        from scipy import stats

        data1 = self.results_df[
            (self.results_df["exp_id"] == exp_id_1) & (self.results_df["dataset"] == dataset)
        ][metric].values

        data2 = self.results_df[
            (self.results_df["exp_id"] == exp_id_2) & (self.results_df["dataset"] == dataset)
        ][metric].values

        t_stat, p_value = stats.ttest_rel(data1, data2)
        pooled_std = np.sqrt((data1.std() ** 2 + data2.std() ** 2) / 2)

        return {
            "exp_1": exp_id_1,
            "exp_2": exp_id_2,
            "dataset": dataset,
            "metric": metric,
            "mean_diff": data2.mean() - data1.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "effect_size_cohen_d": (data2.mean() - data1.mean()) / pooled_std if pooled_std else 0.0,
        }
