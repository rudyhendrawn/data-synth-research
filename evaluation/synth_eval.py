from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


def extract_synthetic_tail(
    X_real: pd.DataFrame | np.ndarray,
    X_resampled: pd.DataFrame | np.ndarray,
    y_real: Optional[pd.Series | np.ndarray] = None,
    y_resampled: Optional[pd.Series | np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract synthetic rows by taking the tail portion of resampled data.
    This assumes resamplers append synthetic samples at the end.
    Returns (X_syn, y_syn) or (None, None) if no synthetic rows.
    """
    X_real_arr = X_real.values if isinstance(X_real, pd.DataFrame) else np.asarray(X_real)
    X_res_arr = X_resampled.values if isinstance(X_resampled, pd.DataFrame) else np.asarray(X_resampled)

    if X_res_arr.shape[0] <= X_real_arr.shape[0]:
        return None, None

    n_syn = X_res_arr.shape[0] - X_real_arr.shape[0]
    X_syn = X_res_arr[-n_syn:]

    if y_real is None or y_resampled is None:
        return X_syn, None

    y_res_arr = y_resampled.values if isinstance(y_resampled, pd.Series) else np.asarray(y_resampled)
    y_syn = y_res_arr[-n_syn:]
    return X_syn, y_syn


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    data = np.sort(np.concatenate([x_sorted, y_sorted]))
    cdf_x = np.searchsorted(x_sorted, data, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, data, side="right") / y_sorted.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def ks_mean(real: np.ndarray, syn: np.ndarray) -> float:
    stats = []
    for i in range(real.shape[1]):
        stats.append(ks_statistic(real[:, i], syn[:, i]))
    stats = [s for s in stats if np.isfinite(s)]
    return float(np.mean(stats)) if stats else float("nan")


def correlation_gap(real: np.ndarray, syn: np.ndarray) -> float:
    real_corr = np.corrcoef(real, rowvar=False)
    syn_corr = np.corrcoef(syn, rowvar=False)
    real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
    syn_corr = np.nan_to_num(syn_corr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(real_corr - syn_corr, ord="fro"))


def duplicate_rate(syn: np.ndarray, decimals: int = 6) -> float:
    if syn.size == 0:
        return float("nan")
    df = pd.DataFrame(np.round(syn, decimals=decimals))
    dup_count = df.duplicated().sum()
    return float(dup_count / len(df))


def tstr_metrics(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    X_real: Optional[np.ndarray] = None,
    y_real: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    X_train_tstr, y_train_tstr, valid = _build_tstr_training(
        X_syn, y_syn, X_real=X_real, y_real=y_real
    )
    if not valid:
        return {
            "tstr_pr_auc": float("nan"),
            "tstr_precision": float("nan"),
            "tstr_recall": float("nan"),
            "tstr_f1": float("nan"),
            "tstr_valid": 0.0,
        }
    clf = LogisticRegression(max_iter=200, random_state=seed)
    clf.fit(X_train_tstr, y_train_tstr)
    probs = clf.predict_proba(X_test)[:, 1]
    pred = (probs >= 0.5).astype(int)
    return {
        "tstr_pr_auc": float(average_precision_score(y_test, probs)),
        "tstr_precision": float(precision_score(y_test, pred, zero_division=0)),
        "tstr_recall": float(recall_score(y_test, pred, zero_division=0)),
        "tstr_f1": float(f1_score(y_test, pred, zero_division=0)),
        "tstr_valid": 1.0,
    }


def _build_tstr_training(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    X_real: Optional[np.ndarray],
    y_real: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    unique_syn = np.unique(y_syn)
    if unique_syn.size >= 2:
        return X_syn, y_syn, True

    if X_real is None or y_real is None:
        return X_syn, y_syn, False

    unique_real = np.unique(y_real)
    missing = [cls for cls in unique_real if cls not in unique_syn]
    if not missing:
        return X_syn, y_syn, False

    majority_class = missing[0]
    majority_mask = y_real == majority_class
    X_majority = X_real[majority_mask]
    y_majority = y_real[majority_mask]

    X_train = np.concatenate([X_majority, X_syn], axis=0)
    y_train = np.concatenate([y_majority, y_syn], axis=0)
    return X_train, y_train, True


def evaluate_synthetic_data(
    X_real: pd.DataFrame | np.ndarray,
    X_syn: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    y_syn: Optional[np.ndarray] = None,
    y_real: Optional[pd.Series | np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, float]:
    X_real_arr = X_real.values if isinstance(X_real, pd.DataFrame) else np.asarray(X_real)
    X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
    y_test_arr = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)

    metrics = {
        "ks_mean": ks_mean(X_real_arr, X_syn),
        "correlation_gap": correlation_gap(X_real_arr, X_syn),
        "duplicate_rate": duplicate_rate(X_syn),
    }

    if y_syn is not None:
        y_real_arr = None
        if y_real is not None:
            y_real_arr = y_real.values if isinstance(y_real, pd.Series) else np.asarray(y_real)
        metrics.update(
            tstr_metrics(
                X_syn,
                y_syn,
                X_test_arr,
                y_test_arr,
                seed=seed,
                X_real=X_real_arr,
                y_real=y_real_arr,
            )
        )

    return metrics
