import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DatasetPreprocessor:
    """
    Unified preprocessor with dataset-specific configurations.
    
    Args:
        config (dict): Dataset-specific configuration dictionary (DATASET_CONFIG).
        verbose (bool): Whether to log detailed processing steps.
    """
    def __init__(self, config: dict, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.label_col = config['label_col']
        self.numeric_cols = []
        self.categorical_cols = config.get('categorical_cols', [])
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def detect_column(self, df: pd.DataFrame):
        """
        Auto-detect numeric and columns if not specified.
        """
        # Get explicitly defined categorical columns
        explicit_cat = set(self.categorical_cols)

        # Find all numeric columns
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove label and drop columns
        drop_cols = set(self.config.get('drop_cols', []))
        all_numeric = [col for col in all_numeric if col != self.label_col and col not in drop_cols]

        # Remaining categorical cols (not in explicit list)
        all_object = df.select_dtypes(include=['object']).columns.tolist()
        all_object = [col for col in all_object if col not in explicit_cat and col != self.label_col and col not in drop_cols]

        self.numeric_cols = all_numeric
        self.categorical_cols = explicit_cat.union(set(all_object))

        if self.verbose:
            logger.info(f"Detected {len(self.numeric_cols)} numeric columns: {self.numeric_cols[:5]}...")
            logger.info(f"Detected {len(self.categorical_cols)} categorical columns: {list(self.categorical_cols)[:5]}...")

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Preprocess data with automatic type handling.

        Args:
            df (pd.DataFrame): Input DataFrame to preprocess.
            fit: Whether to fit the scalers/encoders on this data (True for training data).

        Returns:
            X: Preprocessed features DataFrame.
            y: Labels (or None of label_col not in df).
        """
        # Drop unwanted columns
        drop_cols = self.config.get('drop_cols', [])
        df = df.drop(columns=drop_cols, axis=1, errors='ignore')

        # Detect columns if not done yet
        if not self.numeric_cols and not self.categorical_cols:
            self.detect_column(df)

        # Separate features and labels
        if self.label_col in df.columns:
            y = df[self.label_col].copy()
            X = df.drop(self.label_col, axis=1).copy()
        else:
            y = None
            X = df.copy()

        # Encode categorical columns
        for col in self.categorical_cols:
            if col in X.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in self.label_encoders:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    else:
                        logger.warning(f"LabelEncoder for '{col}' not fitted. Skipping encoding.")

        # Scale numeric columns
        numeric_cols_present = [col for col in self.numeric_cols if col in X.columns]
        if numeric_cols_present:
            if fit:
                X[numeric_cols_present] = self.scaler.fit_transform(X[numeric_cols_present])
            else:
                X[numeric_cols_present] = self.scaler.transform(X[numeric_cols_present])

        if self.verbose:
            logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape if y is not None else 'N/A'}")

        return X, y
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names after preprocessing.
        
        Returns:
            list: List of feature names.
        """
        return self.numeric_cols + list(self.categorical_cols)