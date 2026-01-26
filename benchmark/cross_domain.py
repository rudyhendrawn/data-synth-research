import logging
import os
import random
import time
from typing import (
    Any, 
    Dict, 
    Iterable, 
    List, 
    Optional, 
    Tuple, 
    cast
)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from loader.data_loader import UniversalDataLoader
from preprocessor.data_preprocessor import DatasetPreprocessor
from model.anomaly import add_anomaly_scores

from evaluation.calibration import TemperatureScaler
from evaluation.metrics import (
    bootstrap_pr_auc,
    lift_at_top_k,
    metrics_at_threshold,
    recall_at_precision_target,
    select_threshold_by_fpr,
    select_threshold_by_precision,
)
from evaluation.synth_eval import evaluate_synthetic_data, extract_synthetic_tail
from .ablation import AblationExperiment
from .temporal_cv import TemporalLeakageFreeCV, TimeGapConfig

logger = logging.getLogger(__name__)


class CrossDomainBenchmark:
    """
    Execute ablation experiments across multiple datasets with leakage-safe splits.
    """

    def __init__(
        self,
        datasets: Dict[str, str],
        output_dir: str = "results/cross_domain",
        random_seeds: Optional[List[int]] = None,
        data_root: Optional[str] = None,
        handle_unknown_categories: bool = True,
        precision_target: float = 0.9,
        fpr_target: float = 0.05,
        threshold_strategy: str = "precision",
        bootstrap_samples: int = 2000,
        gap_configs: Optional[Dict[str, TimeGapConfig]] = None,
    ) -> None:
        self.datasets = datasets
        self.output_dir = output_dir
        self.random_seeds = random_seeds or [42, 123]
        self.data_root = data_root
        self.handle_unknown_categories = handle_unknown_categories
        self.precision_target = precision_target
        self.fpr_target = fpr_target
        self.threshold_strategy = threshold_strategy
        self.bootstrap_samples = bootstrap_samples
        os.makedirs(output_dir, exist_ok=True)

        if gap_configs is None:
            defaults = {
                "01_creditcard.csv": TimeGapConfig(lookback_days=7, label_lag_days=14),
                "03_fraud_oracle.csv": TimeGapConfig(lookback_days=30, label_lag_days=45),
                "04_bank_account.csv": TimeGapConfig(lookback_days=30, label_lag_days=30),
                "05_online_payment.csv": TimeGapConfig(lookback_days=1, label_lag_days=1),
            }
            gap_configs = {}
            for filename in datasets.values():
                gap_configs[filename] = defaults.get(filename, TimeGapConfig())

        self.gap_configs = gap_configs

    def run_single_experiment(
        self,
        dataset_name: str,
        experiment: AblationExperiment,
        seed: int,
        data_loader: UniversalDataLoader,
        temporal_cv: Optional[TemporalLeakageFreeCV],
    ) -> Dict[str, Any]:
        """
        Execute single experiment on one dataset with one random seed.
        """
        random.seed(seed)
        np.random.seed(seed)

        df = data_loader.load_raw()
        config = data_loader.config

        train_df, val_df, test_df = self._split_dataset(df, config, temporal_cv)

        if self.handle_unknown_categories:
            train_df, val_df, test_df = self._prepare_unknown_categories(
                train_df,
                val_df,
                test_df,
                config=config,
            )

        preprocessor = DatasetPreprocessor(config, verbose=False)
        X_train, y_train = preprocessor.preprocess(train_df, fit=True)
        X_val, y_val = preprocessor.preprocess(val_df, fit=False)
        X_test, y_test = preprocessor.preprocess(test_df, fit=False)

        X_train = cast(pd.DataFrame, X_train)
        X_val = cast(pd.DataFrame, X_val)
        X_test = cast(pd.DataFrame, X_test)
        y_train = cast(pd.Series, y_train)
        y_val = cast(pd.Series, y_val)
        y_test = cast(pd.Series, y_test)

        anomaly_method = experiment.components.get("anomaly_signal", "None")
        if anomaly_method != "None":
            X_train, X_val, X_test = add_anomaly_scores(
                X_train,
                X_val,
                X_test,
                method=anomaly_method,
                random_state=seed,
            )
            X_train = cast(pd.DataFrame, X_train)
            X_val = cast(pd.DataFrame, X_val)
            X_test = cast(pd.DataFrame, X_test)

        assert X_test is not None
        assert X_val is not None
        X_train_resampled, y_train_resampled = self._apply_oversampling(
            X_train, y_train, method=experiment.components["oversampling"], seed=seed
        )
        syn_eval = {}
        syn_X, syn_y = extract_synthetic_tail(X_train, X_train_resampled, y_train, y_train_resampled)
        if syn_X is not None:
            syn_eval = evaluate_synthetic_data(
                X_real=X_train,
                X_syn=syn_X,
                X_test=X_test,
                y_test=y_test,
                y_syn=syn_y,
                y_real=y_train,
                seed=seed,
            )

        model = self._get_model(
            experiment.components["model"],
            seed=seed,
            scale_pos_weight=self._compute_scale_pos_weight(y_train_resampled),
        )

        start_time = time.time()
        model.fit(X_train_resampled, y_train_resampled)
        train_time = time.time() - start_time

        calib_method = experiment.components["calibration"]
        if calib_method in {"Platt", "Isotonic"}:
            model = self._apply_calibration(model, X_val, y_val, calib_method)
            calib_probs = np.asarray(model.predict_proba(X_val))[:, 1]
            test_probs = np.asarray(model.predict_proba(X_test))[:, 1]
        elif calib_method == "Temperature":
            raw_calib_probs = np.asarray(model.predict_proba(X_val))[:, 1]
            temp_scaler = TemperatureScaler()
            temp_scaler.fit(raw_calib_probs, y_val)
            calib_probs = temp_scaler.transform(raw_calib_probs)
            test_probs = temp_scaler.transform(np.asarray(model.predict_proba(X_test))[:, 1])
        else:
            calib_probs = np.asarray(model.predict_proba(X_val))[:, 1]
            test_probs = np.asarray(model.predict_proba(X_test))[:, 1]

        if self.threshold_strategy == "fpr":
            threshold, calib_prec, calib_rec, calib_fpr = select_threshold_by_fpr(
                y_val, calib_probs, self.fpr_target
            )
        else:
            threshold, calib_prec, calib_rec = select_threshold_by_precision(
                y_val, calib_probs, self.precision_target
            )
            calib_fpr = metrics_at_threshold(y_val, calib_probs, threshold)["fpr"]

        test_metrics_at_th = metrics_at_threshold(y_test, test_probs, threshold)
        test_pred = (test_probs >= threshold).astype(int)

        metrics = {
            "dataset": dataset_name,
            "exp_id": experiment.exp_id,
            "seed": seed,
            "oversampling": experiment.components["oversampling"],
            "anomaly_signal": anomaly_method,
            "model": experiment.components["model"],
            "calibration": calib_method,
            "components": str(experiment.components),
            "threshold_strategy": self.threshold_strategy,
            "precision_target": self.precision_target,
            "fpr_target": self.fpr_target,
            "threshold": threshold,
            "calib_precision_at_threshold": calib_prec,
            "calib_recall_at_threshold": calib_rec,
            "calib_fpr_at_threshold": calib_fpr,
            "pr_auc": average_precision_score(y_test, test_probs),
            "precision": precision_score(y_test, test_pred, zero_division=0),
            "recall": recall_score(y_test, test_pred, zero_division=0),
            "f1": test_metrics_at_th["f1"],
            "recall_at_precision_target": recall_at_precision_target(
                y_test, test_probs, self.precision_target
            ),
            "lift_top_1pct": lift_at_top_k(y_test, test_probs, 0.01),
            "train_time": train_time,
            "train_samples": len(X_train_resampled),
            "test_samples": len(X_test),
            "train_fraud_ratio": float(np.sum(y_train_resampled) / len(y_train_resampled)),
            "test_fraud_ratio": float(np.sum(y_test) / len(y_test)),
        }
        metrics.update(
            {
                "ks_mean": syn_eval.get("ks_mean"),
                "correlation_gap": syn_eval.get("correlation_gap"),
                "duplicate_rate": syn_eval.get("duplicate_rate"),
                "tstr_pr_auc": syn_eval.get("tstr_pr_auc"),
                "tstr_precision": syn_eval.get("tstr_precision"),
                "tstr_recall": syn_eval.get("tstr_recall"),
                "tstr_f1": syn_eval.get("tstr_f1"),
            }
        )

        pr_auc_ci = bootstrap_pr_auc(
            y_test,
            test_probs,
            n_bootstrap=self.bootstrap_samples,
            seed=seed,
        )
        metrics.update(
            {
                "pr_auc_ci_lower": pr_auc_ci["lower"],
                "pr_auc_ci_upper": pr_auc_ci["upper"],
                "pr_auc_bootstrap_mean": pr_auc_ci["mean"],
                "pr_auc_bootstrap_n": pr_auc_ci["n"],
            }
        )

        return metrics

    def run_ablation_study(
        self,
        experiments: List[AblationExperiment],
        project_root: str,
        experiment_tag: Optional[str] = None,
        dataset_tag: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run ablation study across datasets and seeds.
        """
        all_results: List[Dict[str, Any]] = []
        total_runs = len(self.datasets) * len(experiments) * len(self.random_seeds)
        current_run = 0

        for dataset_label, filename in self.datasets.items():
            logger.info("\n%s", "=" * 60)
            logger.info("Processing dataset: %s (%s)", dataset_label, filename)
            logger.info("%s", "=" * 60)

            loader = UniversalDataLoader(
                filename,
                project_root=project_root,
                data_root=self.data_root or os.path.join(project_root, "data"),
                verbose=False,
            )

            gap_config = self.gap_configs.get(filename, TimeGapConfig())
            temporal_cv = TemporalLeakageFreeCV(gap_config=gap_config)

            for experiment in experiments:
                for seed in self.random_seeds:
                    current_run += 1
                    logger.info(
                        "[%s/%s] Running %s on %s (seed=%s)",
                        current_run,
                        total_runs,
                        experiment.exp_id,
                        dataset_label,
                        seed,
                    )

                    try:
                        metrics = self.run_single_experiment(
                            dataset_label, experiment, seed, loader, temporal_cv
                        )
                        all_results.append(metrics)
                    except Exception as exc:
                        logger.exception("Failed: %s", exc)
                        continue

        results_df = pd.DataFrame(all_results)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        dataset_tag = (dataset_tag or "all_datasets").replace(".csv", "")
        experiment_tag = experiment_tag or "ablation"
        results_path = os.path.join(
            self.output_dir, f"{dataset_tag}_{experiment_tag}_{timestamp}.csv"
        )
        results_df.to_csv(results_path, index=False)
        logger.info("\nResults saved to %s", results_path)

        return results_df

    def _split_dataset(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        temporal_cv: Optional[TemporalLeakageFreeCV],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        label_col = config["label_col"]
        temporal_col = config.get("temporal_col")
        use_temporal = (
            temporal_cv is not None
            and temporal_col is not None
            and temporal_col in df.columns
            and not config.get("shuffle", True)
        )

        if use_temporal:
            assert temporal_cv is not None
            if temporal_col is None:
                raise ValueError("Temporal column is not set but temporal split was requested.")
            return temporal_cv.split_with_gap(df, temporal_col=temporal_col, label_col=label_col)

        test_size = config.get("test_size", 0.4)
        val_size = 0.5
        stratify_col = df[label_col] if config.get("stratify", True) else None

        train_df, temp_df = train_test_split(
            df, test_size=test_size, stratify=stratify_col, random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_size,
            stratify=temp_df[label_col] if stratify_col is not None else None,
            random_state=42,
        )

        return train_df, val_df, test_df

    def _prepare_unknown_categories(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Dict[str, Any],
        unknown_token: str = "__UNK__",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cat_cols = [col for col in config.get("categorical_cols", []) if col in train_df.columns]
        if not cat_cols:
            return train_df, val_df, test_df

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        for col in cat_cols:
            train_df[col] = train_df[col].astype(str)
            val_df[col] = val_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

        known = {col: set(train_df[col].unique()) for col in cat_cols}

        for col in cat_cols:
            if unknown_token not in known[col]:
                known[col].add(unknown_token)

        for col in cat_cols:
            val_df[col] = val_df[col].where(val_df[col].isin(known[col]), other=unknown_token)
            test_df[col] = test_df[col].where(test_df[col].isin(known[col]), other=unknown_token)

        stacked = train_df[cat_cols].stack()
        if unknown_token not in set(pd.unique(stacked.to_numpy())):
            train_df = self._append_unknown_row(train_df, cat_cols, config.get("label_col"), unknown_token)

        return train_df, val_df, test_df

    def _append_unknown_row(
        self,
        train_df: pd.DataFrame,
        cat_cols: Iterable[str],
        label_col: Optional[str],
        unknown_token: str,
    ) -> pd.DataFrame:
        new_row: Dict[str, Any] = {}
        for col in train_df.columns:
            if col in cat_cols:
                new_row[col] = unknown_token
            elif col == label_col:
                if label_col and label_col in train_df.columns:
                    new_row[col] = train_df[label_col].mode().iloc[0]
                else:
                    new_row[col] = 0
            else:
                new_row[col] = 0

        return pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)

    def _apply_oversampling(
        self, X: Any, y: Any, method: str, seed: int
    ) -> Tuple[Any, Any]:
        if method == "None":
            return X, y
        if method == "ROS":
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=seed)
            return _normalize_resample_output(ros.fit_resample(X, y))
        if method == "SMOTE":
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=seed)
            return _normalize_resample_output(sm.fit_resample(X, y))
        if method == "BorderlineSMOTE":
            from imblearn.over_sampling import BorderlineSMOTE

            bsm = BorderlineSMOTE(random_state=seed)
            return _normalize_resample_output(bsm.fit_resample(X, y))
        if method == "SMOTEENN":
            from imblearn.combine import SMOTEENN

            sme = SMOTEENN(random_state=seed)
            return _normalize_resample_output(sme.fit_resample(X, y))
        if method == "PytorchGAN":
            from model import oversample_with_pytorch_gan

            X_res, y_res, _, _ = oversample_with_pytorch_gan(
                X, y, target_class=1, oversample_ratio=1.0, epochs=300, batch_size=128
            )
            return X_res, y_res
        if method == "CTGAN":
            from model import oversample_with_ctgan

            X_res, y_res, _, _ = oversample_with_ctgan(
                X, y, target_class=1, oversample_ratio=1.0, epochs=300, batch_size=128
            )
            return X_res, y_res
        if method == "ConditionalWGAN-GP":
            from model import oversample_with_cond_wgangp

            X_res, y_res, _, _ = oversample_with_cond_wgangp(
                X, y, target_class=1, target_ratio=1.0, epochs=300, batch_size=128
            )
            return X_res, y_res

        raise ValueError(f"Unknown oversampling method: {method}")

    def _get_model(self, model_name: str, seed: int, scale_pos_weight: float):
        if model_name == "XGBoost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
            )
        if model_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=200, random_state=seed)
        if model_name == "DecisionTree":
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(random_state=seed)
        if model_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=200, random_state=seed)
        if model_name == "MLP":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed)

        raise ValueError(f"Unknown model: {model_name}")

    def _apply_calibration(self, model, X_val, y_val, method: str):
        if method == "Platt":
            return CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(X_val, y_val)
        if method == "Isotonic":
            return CalibratedClassifierCV(model, method="isotonic", cv="prefit").fit(X_val, y_val)
        return model

    @staticmethod
    def _compute_scale_pos_weight(y) -> float:
        y_array = np.asarray(y)
        positives = np.sum(y_array == 1)
        negatives = np.sum(y_array == 0)
        if positives == 0:
            return 1.0
        return float(negatives / positives)


def _normalize_resample_output(result: Any) -> Tuple[Any, Any]:
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    raise ValueError("Unexpected resample output format.")
