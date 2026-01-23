import os
import logging
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas._typing import DtypeArg

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from preprocessor.data_config import DATASET_CONFIG
from preprocessor.data_preprocessor import DatasetPreprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class UniversalDataLoader:
    """
    Load and preprocess any dataset in the project.
    """
    def __init__(
            self, 
            dataset_name: str, 
            project_root: str, 
            data_root: str, 
            verbose: bool = True,
            large_data: bool = False
        ):
        """
        Args:
            dataset_name (str): Filename of the dataset to load (e.g., '01_creditcard.csv').
            project_root (str): Root directory of the project.
            verbose (bool): Whether to log detailed processing steps.
            large_data (bool): Whether the dataset is large, affecting loading/preprocessing.
        """
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Dataset '{dataset_name}' not configured. Available: {list(DATASET_CONFIG.keys())}")
        
        self.dataset_name = dataset_name
        self.project_root = project_root
        self.data_root = data_root if data_root is not None else os.path.join(project_root, 'data')
        self.config = DATASET_CONFIG[dataset_name]
        self.verbose = verbose
        self.large_data = bool(large_data) if large_data is not None else False
        self.preprocessor: Optional[DatasetPreprocessor] = None
        self.data_path = os.path.join(self.data_root, dataset_name)

        logger.info(f"Initialized loader for: {self.config['name']}")
        logger.info(f"Data path: {self.data_path}")

    def load_raw(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def load_and_preprocess(self, fit: bool = True) -> tuple:
        """
        Load and preprocess entire dataset.

        Args:
            fit (bool): Whether to fit the preprocessor on this data (True for training data).

        Returns:
            X: Preprocessed features
            y: Labels
            preprocessor: Fitted DatasetPreprocessor instance
        """
        if self.large_data:
            logger.info("Large dataset detected. Using chunked processing.")
            return self.load_and_preprocess_large()
        df = self.load_raw()

        # Initialize preprocessor
        self.preprocessor = DatasetPreprocessor(self.config, verbose=self.verbose)

        # Preprocess data
        X, y = self.preprocessor.preprocess(df, fit=fit)

        logger.info(f"Class distribution after preprocessing:\n{y.value_counts(normalize=True)}")
        
        return X, y, self.preprocessor
    
    def _iter_raw_chunks(
            self,
            chunk_size: int,
            usecols: Optional[List[str]] = None,
            dtype: Optional[DtypeArg] = None
    ) -> Iterator[pd.DataFrame]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")

        if usecols is None and dtype is None:
            return pd.read_csv(self.data_path, chunksize=chunk_size)
        if usecols is None:
            return pd.read_csv(self.data_path, chunksize=chunk_size, dtype=dtype)
        if dtype is None:
            return pd.read_csv(self.data_path, chunksize=chunk_size, usecols=usecols)
        return pd.read_csv(self.data_path, chunksize=chunk_size, usecols=usecols, dtype=dtype)

    def _fit_preprocessor_stream(
            self,
            chunk_size: int,
            fit_rows: int,
            usecols: Optional[List[str]] = None,
            dtype: Optional[DtypeArg] = None
    ) -> None:
        if self.preprocessor is None:
            self.preprocessor = DatasetPreprocessor(self.config, verbose=self.verbose)

        if fit_rows is None or fit_rows <= 0:
            sample_df = next(self._iter_raw_chunks(chunk_size=chunk_size, usecols=usecols, dtype=dtype))
        else:
            collected = []
            seen = 0
            for chunk in self._iter_raw_chunks(chunk_size=chunk_size, usecols=usecols, dtype=dtype):
                collected.append(chunk)
                seen += chunk.shape[0]
                if seen >= fit_rows:
                    break

            if not collected:
                raise ValueError("No data found to fit preprocessor.")
            
            sample_df = pd.concat(collected, ignore_index=True)

            if fit_rows is not None and fit_rows < len(sample_df):
                sample_df = sample_df.sample(n=fit_rows, random_state=42)

        self.preprocessor.preprocess(sample_df, fit=True)

    def iter_preprocessed_batches(
        self,
        chunk_size: int = 10000,
        fit_rows: int = 20000,
        usecols: Optional[List[str]] = None,
        dtype: Optional[DtypeArg] = None
            
    ) -> Iterator[Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """
        Iterate over preprocessed data in chunks.

        Args:
            chunk_size (int): Number of rows per chunk.
            fit_rows (int): Number of rows to use for fitting the preprocessor. 
            usecols (list): Columns to read from CSV.
            dtype (dict): Data types for columns.
        Returns:
            Iterator of (X_batch, y_batch) tuples.
        """
        if not self.large_data:
            raise ValueError("Large data mode is disabled. Set large_data=True to use this method.")
        
        if self.preprocessor is None:
            self._fit_preprocessor_stream(
                chunk_size=chunk_size,
                fit_rows=fit_rows,
                usecols=usecols,
                dtype=dtype
            )

        assert self.preprocessor is not None
        for chunk in self._iter_raw_chunks(chunk_size=chunk_size, usecols=usecols, dtype=dtype):
            yield self.preprocessor.preprocess(chunk, fit=False)

    def load_and_preprocess_large(
            self,
            chunk_size: int = 10000,
            fit_rows: int = 20000,
            usecols: Optional[List[str]] = None,
            dtype: Optional[DtypeArg] = None
    ) -> Tuple[Iterator[Tuple[pd.DataFrame, Optional[pd.Series]]], DatasetPreprocessor]:
        """
        Load and preprocess large dataset in chunks.

        Args:
            chunk_size (int): Number of rows per chunk.
            fit_rows (int): Number of rows to use for fitting the preprocessor.
            usecols (list): Columns to read from CSV.
            dtype (dict): Data types for columns.
        Returns:
            data_iter: Iterator of (X_batch, y_batch) tuples.
            preprocessor: Fitted DatasetPreprocessor instance
        """
        if not self.large_data:
            raise ValueError("Large data mode is disabled. Set large_data=True to use this method.")
        
        logger.info("Fitting preprocessor on large data...")
        self._fit_preprocessor_stream(
            chunk_size=chunk_size,
            fit_rows=fit_rows,
            usecols=usecols,
            dtype=dtype
        )

        data_iter = self.iter_preprocessed_batches(
            chunk_size=chunk_size,
            fit_rows=fit_rows,
            usecols=usecols,
            dtype=dtype
        )

        assert self.preprocessor is not None
        logger.info("Preprocessor fitted for large data.")

        return data_iter, self.preprocessor


    def train_val_test_split(self) -> tuple:
        """
        Load and split the dataset into train, validation, and test sets.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
        """
        df = self.load_raw()

        # Initialize preprocessor
        self.preprocessor = DatasetPreprocessor(self.config, verbose=self.verbose)

        # Handle temporal data (time-aware split)
        if self.config['type'] in ['numeric_temporal', 'mixed'] and not self.config.get('shuffle', True):
            logger.info("Using time-aware split (forward-chaining) based on temporal column.")
            
            # Sort by temporal column if exists
            temporal_col = self.config.get('temporal_col')
            if temporal_col and temporal_col in df.columns:
                df = df.sort_values(by=temporal_col).reset_index(drop=True)
                logger.info(f"Sorted by temporal column: {temporal_col}")

            # Split into train (60%), val (20%), test (20%)
            test_size = self.config['test_size'] # 0.4
            val_size = 0.5 # Half of test (which is 40%) -> 20% of total

            train, temp = train_test_split(df, test_size=test_size, shuffle=False)
            val, test = train_test_split(temp, test_size=val_size, shuffle=False)
        else:
            # Stratified Shuffle Split
            logger.info("Using stratified shuffle split.")

            test_size = self.config['test_size'] # 0.4
            val_size = 0.5 # Half of test (which is 40%) -> 20% of total
            label_col = self.config['label_col']

            if self.config.get('stratify', True):
                stratify_col = df[label_col]
            else:
                stratify_col = None

            train, temp = train_test_split(
                df, 
                test_size=test_size,
                stratify=stratify_col,
                random_state=42
            )

            val, test = train_test_split(
                temp,
                test_size=val_size,
                stratify=temp[label_col] if stratify_col is not None else None,
                random_state=42
            )

        # Preprocess all splits
        X_train, y_train = self.preprocessor.preprocess(train, fit=True)
        X_val, y_val = self.preprocessor.preprocess(val, fit=False)
        X_test, y_test = self.preprocessor.preprocess(test, fit=False)

        # Log info
        logger.info(f"\nData split:")
        logger.info(f". Train: {X_train.shape[0]} samples | Class distribution: {y_train.value_counts(normalize=True).to_dict()}")
        logger.info(f". Validation: {X_val.shape[0]} samples | Class distribution: {y_val.value_counts(normalize=True).to_dict()}")
        logger.info(f". Test: {X_test.shape[0]} samples | Class distribution: {y_test.value_counts(normalize=True).to_dict()}")

        return X_train, X_val, X_test, y_train, y_val, y_test, self.preprocessor
        
