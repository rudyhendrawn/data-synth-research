import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimeGapConfig:
    """Configuration for temporal gaps to prevent leakage."""
    lookback_days: int = 30
    label_lag_days: int = 45
    gap_days: int = 45

    def __post_init__(self) -> None:
        self.gap_days = max(self.lookback_days, self.label_lag_days)


class TemporalLeakageFreeCV:
    """
    Temporal cross-validation with purging and embargo to prevent leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap_config: Optional[TimeGapConfig] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ) -> None:
        self.n_splits = n_splits
        self.gap_config = gap_config or TimeGapConfig()
        self.test_size = test_size
        self.val_size = val_size

    def split_with_gap(
        self,
        df: pd.DataFrame,
        temporal_col: str,
        label_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data with temporal gap (purging + embargo).
        """
        if temporal_col not in df.columns:
            raise ValueError(f"Temporal column '{temporal_col}' not found in dataframe.")

        df = df.sort_values(temporal_col).reset_index(drop=True)
        n = len(df)
        test_start_idx = int(n * (1 - self.test_size))
        val_start_idx = int(n * (1 - self.test_size - self.val_size))

        gap_label = None
        gap_rows = 0

        if pd.api.types.is_datetime64_any_dtype(df[temporal_col]):
            gap_timedelta = pd.Timedelta(days=self.gap_config.gap_days)
            gap_label = gap_timedelta

            train_end_time = df.iloc[val_start_idx][temporal_col] - gap_timedelta
            val_end_time = df.iloc[test_start_idx][temporal_col] - gap_timedelta

            train = df[df[temporal_col] <= train_end_time].copy()
            val = df[
                (df[temporal_col] > train_end_time + gap_timedelta)
                & (df[temporal_col] <= val_end_time)
            ].copy()
            test = df[df[temporal_col] > val_end_time + gap_timedelta].copy()
        else:
            gap_rows = int(len(df) * (self.gap_config.gap_days / 365))
            gap_label = gap_rows

            train = df[: max(0, val_start_idx - gap_rows)].copy()
            val = df[val_start_idx : max(val_start_idx, test_start_idx - gap_rows)].copy()
            test = df[test_start_idx:].copy()

        if label_col in df.columns:
            logger.info("Split with gap applied:")
            logger.info("  Train: %s samples (%s)", len(train), train[label_col].value_counts().to_dict())
            logger.info("  Gap 1: %s", gap_label)
            logger.info("  Val:   %s samples (%s)", len(val), val[label_col].value_counts().to_dict())
            logger.info("  Gap 2: %s", gap_label)
            logger.info("  Test:  %s samples (%s)", len(test), test[label_col].value_counts().to_dict())

        return train, val, test

    def forward_chaining_cv(
        self,
        df: pd.DataFrame,
        temporal_col: str,
        label_col: str,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate forward-chaining CV folds with row-based gaps.
        """
        if temporal_col not in df.columns:
            raise ValueError(f"Temporal column '{temporal_col}' not found in dataframe.")

        df = df.sort_values(temporal_col).reset_index(drop=True)
        n = len(df)

        fold_size = n // (self.n_splits + 1)
        gap_rows = int(fold_size * (self.gap_config.gap_days / 365))

        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            val_start = train_end + gap_rows
            val_end = val_start + fold_size

            if val_end > n:
                break

            train_idx = np.arange(0, max(0, train_end - gap_rows))
            val_idx = np.arange(val_start, min(val_end, n))
            folds.append((train_idx, val_idx))

        logger.info("Generated %s forward-chaining folds with gaps", len(folds))
        return folds
