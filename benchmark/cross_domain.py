import logging
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from loader.data_loader import UniversalDataLoader
from preprocessor.data_preprocessor import DatasetPreprocessor

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
    ) -> None:
        self.datasets = datasets
        self.output_dir = output_dir
        self.random_seeds = random_seeds or [42, 123, 456, 789, 1011]
        self.data_root = data_root
        self.handle_unknown_categories = handle_unknown_categories
        os.makedirs(output_dir, exist_ok=True)

        self.gap_configs = {
            "01_creditcard.csv": TimeGapConfig(lookback_days=7, label_lag_days=14),
            "03_fraud_oracle.csv": TimeGapConfig(lookback_days=30, label_lag_days=45),
            "04_bank_account.csv": TimeGapConfig(lookback_days=30, label_lag_days=30),
            "05_online_payment.csv": TimeGapConfig(lookback_days=1, label_lag_days=1),
        }

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

        X_train_resampled, y_train_resampled = self._apply_oversampling(
            X_train, y_train, method=experiment.components["oversampling"], seed=seed
        )

        model = self._get_model(
            experiment.components["model"],
            seed=seed,
            scale_pos_weight=self._compute_scale_pos_weight(y_train_resampled),
        )

        start_time = time.time()
        model.fit(X_train_resampled, y_train_resampled)
        train_time = time.time() - start_time

        if experiment.components["calibration"] != "None":
            model = self._apply_calibration(model, X_val, y_val, experiment.components["calibration"])

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "dataset": dataset_name,
            "exp_id": experiment.exp_id,
            "seed": seed,
            "oversampling": experiment.components["oversampling"],
            "model": experiment.components["model"],
            "calibration": experiment.components["calibration"],
            "pr_auc": average_precision_score(y_test, y_pred_proba),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_time": train_time,
            "train_samples": len(X_train_resampled),
            "test_samples": len(X_test),
            "train_fraud_ratio": float(np.sum(y_train_resampled) / len(y_train_resampled)),
            "test_fraud_ratio": float(np.sum(y_test) / len(y_test)),
        }

        return metrics

    def run_ablation_study(self, experiments: List[AblationExperiment], project_root: str) -> pd.DataFrame:
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
        results_path = os.path.join(self.output_dir, f"ablation_results_{timestamp}.csv")
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

        if unknown_token not in set(train_df[cat_cols].stack().unique()):
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

    def _apply_oversampling(self, X, y, method: str, seed: int):
        if method == "None":
            return X, y
        if method == "ROS":
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=seed)
            return ros.fit_resample(X, y)
        if method == "SMOTE":
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=seed)
            return sm.fit_resample(X, y)
        if method == "BorderlineSMOTE":
            from imblearn.over_sampling import BorderlineSMOTE

            bsm = BorderlineSMOTE(random_state=seed)
            return bsm.fit_resample(X, y)
        if method == "SMOTEENN":
            from imblearn.combine import SMOTEENN

            sme = SMOTEENN(random_state=seed)
            return sme.fit_resample(X, y)
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
