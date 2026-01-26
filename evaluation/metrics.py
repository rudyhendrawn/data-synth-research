from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, 
    f1_score, 
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    roc_curve
)


def lift_at_top_k(y_true, probs, top_k_percent: float) -> float:
    y_arr = y_true.values if isinstance(y_true, pd.Series) else np.asarray(y_true)
    k = int(len(y_arr) * top_k_percent)
    if k <= 0:
        k = 1
    top_k_indices = np.argsort(probs)[-k:]
    top_k_actual = y_arr[top_k_indices]
    fraud_rate_top_k = np.sum(top_k_actual == 1) / k
    fraud_rate_overall = np.sum(y_arr == 1) / len(y_arr)
    if fraud_rate_overall == 0:
        return 0.0
    return fraud_rate_top_k / fraud_rate_overall


def recall_at_precision_target(y_true, probs, precision_target: float) -> float:
    prec, rec, _ = precision_recall_curve(y_true, probs)
    if len(prec) <= 1:
        return 0.0
    prec_thr = prec[1:]
    rec_thr = rec[1:]
    mask = prec_thr >= precision_target
    if not np.any(mask):
        return 0.0
    return float(rec_thr[mask].max())


def select_threshold_by_precision(
    y_true,
    probs,
    precision_target: float,
) -> Tuple[float, float, float]:
    prec, rec, thr = precision_recall_curve(y_true, probs)
    if len(thr) == 0:
        return 0.5, 0.0, 0.0
    prec_thr = prec[1:]
    rec_thr = rec[1:]
    mask = prec_thr >= precision_target
    if np.any(mask):
        best_idx = np.argmax(rec_thr[mask])
        candidate_thresholds = thr[mask]
        candidate_prec = prec_thr[mask]
        candidate_rec = rec_thr[mask]
        return (
            float(candidate_thresholds[best_idx]),
            float(candidate_prec[best_idx]),
            float(candidate_rec[best_idx]),
        )
    best_idx = np.argmax(prec_thr)
    return float(thr[best_idx]), float(prec_thr[best_idx]), float(rec_thr[best_idx])


def select_threshold_by_fpr(
    y_true,
    probs,
    fpr_target: float,
) -> Tuple[float, float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, probs)
    if len(thr) == 0:
        return 0.5, 0.0, 0.0, 0.0
    mask = fpr <= fpr_target
    if np.any(mask):
        best_idx = np.argmax(tpr[mask])
        threshold = float(thr[mask][best_idx])
    else:
        best_idx = np.argmin(fpr)
        threshold = float(thr[best_idx])

    pred = (probs >= threshold).astype(int)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    tn = np.sum((y_true == 0) & (pred == 0))
    fp = np.sum((y_true == 0) & (pred == 1))
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return threshold, float(precision), float(recall), float(fpr_value)


def metrics_at_threshold(y_true, probs, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    tn = np.sum((y_true == 0) & (pred == 0))
    fp = np.sum((y_true == 0) & (pred == 1))
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr_value),
    }


def bootstrap_pr_auc(
    y_true,
    probs,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y_arr = y_true.values if isinstance(y_true, pd.Series) else np.asarray(y_true)
    probs_arr = np.asarray(probs)

    scores = []
    n = len(y_arr)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_sample = np.asarray(y_arr[idx])
        if len(np.unique(y_sample)) < 2:
            continue
        score = average_precision_score(y_sample, probs_arr[idx])
        scores.append(score)

    if not scores:
        return {
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "n": 0,
        }

    scores_arr = np.asarray(scores)
    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q
    return {
        "mean": float(np.mean(scores_arr)),
        "lower": float(np.quantile(scores_arr, lower_q)),
        "upper": float(np.quantile(scores_arr, upper_q)),
        "n": float(len(scores_arr)),
    }
