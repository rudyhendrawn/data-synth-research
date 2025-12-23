import os
import sys
import logging
import subprocess
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basic_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model.sdv_model import oversample_with_sdv_ctgan, oversample_with_copula_gan
from evaluation.evaluation import evaluate_models_to_dataframe

try:
    subprocess.run(['brew', 'install', 'libomp'], check=True, capture_output=True)
    logger.info("OpenMP runtime installed successfully")
except subprocess.CalledProcessError:
    logger.warning("Failed to install OpenMP runtime. Please run 'brew install libomp' manually in terminal")
except FileNotFoundError:
    logger.warning("Homebrew not found. Please install Homebrew first or run 'brew install libomp' manually")

# Load data
data_path = os.path.join(project_root, 'data', '01_creditcard.csv')
logger.info(f"Loading data from {data_path}")
df = pd.read_csv(data_path)
logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
random_state = 42

# Preprocessing
logger.info("Starting preprocessing")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Split the data
df = df.sort_values(by='Time').reset_index(drop=True)
df = df.drop(['Time'], axis=1)

train, temp = train_test_split(df, test_size=0.4, shuffle=False)
val, test = train_test_split(temp, test_size=0.5, shuffle=False)

X_train, y_train = train.drop('Class', axis=1), train['Class']
X_val, y_val = val.drop('Class', axis=1), val['Class']
X_test, y_test = test.drop('Class', axis=1), test['Class']

logger.info(f"Train set: {X_train.shape[0]} samples")
logger.info(f"Validation set: {X_val.shape[0]} samples")
logger.info(f"Test set: {X_test.shape[0]} samples")
logger.info(f"Class distribution before resampling: {y_train.value_counts().to_dict()}")


# 1. SDV GAN
logger.info("Training SDV CTGAN")
X_train_sdv_gan, y_train_sdv_gan = oversample_with_sdv_ctgan(
    X_train, y_train, target_class=1, oversample_ratio=1.0, epochs=500, batch_size=250
)
logger.info(f"SDV GAN - Class distribution: {pd.Series(y_train_sdv_gan).value_counts().to_dict()}")

# 2. SDV Copula GAN
logger.info("Training SDV Copula GAN")
X_train_sdv_copula_gan, y_train_sdv_copula_gan = oversample_with_copula_gan(
    X_train, y_train, target_class=1, oversample_ratio=1.0, epochs=500, batch_size=250
)
logger.info(f"SDV Copula GAN - Class distribution: {pd.Series(y_train_sdv_copula_gan).value_counts().to_dict()}")

# Training and evaluation
params = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (len(y_train)/y_train.sum()),
}

training_data = {
    'XGB with SDV CTGAN': (X_train_sdv_gan, y_train_sdv_gan),
    'XGB with SDV Copula GAN': (X_train_sdv_copula_gan, y_train_sdv_copula_gan),
}

logger.info("Training models")
models_dict = {}
for model_name, (X, y) in training_data.items():
    logger.info(f"Training {model_name}")
    model = XGBClassifier(**params, use_label_encoder=False)
    model.fit(X, y)
    models_dict[model_name] = model

# Evaluate models
logger.info("Evaluating models")
results_df = evaluate_models_to_dataframe(models_dict, X_test, y_test)

# Load  results
results_path = os.path.join(project_root, 'results', 'xgb_imbalanced_handling_results.csv')
previous_results_df = pd.read_csv(results_path)
results_df = pd.concat([previous_results_df, results_df], ignore_index=True)
results_df.to_csv(results_path, index=False)

# results_df.to_csv(results_path, index=False)
logger.info(f"Results saved to {results_path}")
