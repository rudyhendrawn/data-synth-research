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
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from model import oversample_with_pytorch_gan, oversample_with_ctgan, oversample_with_cond_wgangp
from model.anomaly import add_anomaly_scores
from evaluation.evaluation import evaluate_models_to_dataframe
from evaluation.synth_eval import evaluate_synthetic_data, extract_synthetic_tail
from loader.data_loader import UniversalDataLoader
from preprocessor.data_config import DATASET_CONFIG

try:
    subprocess.run(['brew', 'install', 'libomp'], check=True, capture_output=True)
    logger.info("OpenMP runtime installed successfully")
except subprocess.CalledProcessError:
    logger.warning("Failed to install OpenMP runtime")
except FileNotFoundError:
    logger.warning("Homebrew not found")

# === DATASET SELECTION ===
# Change this to switch datasets
DATASET_NAME = '05_online_payment.csv' # or '01_creditcard.csv', '03_fraud_oracle.csv', '04_bank_account.csv', '05_online_payment.csv'
DATA_ROOT = "/Users/rudyhendrawan/Projects/data"
logger.info(f"Loading dataset: {DATASET_CONFIG[DATASET_NAME]['name']}")

# === ANOMALY FEATURE SETTINGS ===
USE_ANOMALY_FEATURES = True
ANOMALY_METHOD = "IsolationForest"  # None, IsolationForest, LOF, Autoencoder
ANOMALY_CONTAMINATION = 0.01

# === LOAD AND PREPROCESS DATA ===
loader = UniversalDataLoader(
    DATASET_NAME, 
    project_root=project_root, 
    data_root=DATA_ROOT,
    verbose=True,
    large_data=False # For large datasets e.g. 04_bank_account.csv, use chunking
)

X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = loader.train_val_test_split()
logger.info(f"Data Training: {X_train.shape}, {y_train.shape}")
logger.info(f"Data Validation: {X_val.shape}, {y_val.shape}")
logger.info(f"Data Testing: {X_test.shape}, {y_test.shape}")
random_state = 42

if USE_ANOMALY_FEATURES and ANOMALY_METHOD != "None":
    logger.info("Adding anomaly score feature: %s", ANOMALY_METHOD)
    X_train, X_val, X_test = add_anomaly_scores(
        X_train,
        X_val,
        X_test,
        method=ANOMALY_METHOD,
        random_state=random_state,
        contamination=ANOMALY_CONTAMINATION,
    )

# === RESAMPLING METHODS ===
logger.info("Applying resampling techniques")

# 1. Regular SMOTE
logger.info("Applying SMOTE")
sm = SMOTE(random_state=random_state)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train) # pyright: ignore[reportAssignmentType]
logger.info(f"SMOTE - Class distribution: {y_train_smote.value_counts().to_dict()}")

# 2. Borderline SMOTE
logger.info("Applying Borderline SMOTE")
blsmote = BorderlineSMOTE(random_state=random_state)
X_train_blsmote, y_train_blsmote, *_ = blsmote.fit_resample(X_train, y_train)
logger.info(f"Borderline SMOTE - Class distribution: {y_train_blsmote.value_counts().to_dict()}")

# 3. SMOTEENN
logger.info("Applying SMOTEENN")
smote_enn = SMOTEENN(random_state=random_state)
X_train_smoteenn, y_train_smoteenn, *_ = smote_enn.fit_resample(X_train, y_train)
logger.info(f"SMOTEENN - Class distribution: {y_train_smoteenn.value_counts().to_dict()}")

epochs = 500
batch_size = 64

# 4. PyTorch GAN
logger.info("Training PyTorch GAN")
X_train_gan, y_train_gan, gen_losses, disc_losses = oversample_with_pytorch_gan(
    X_train, y_train, target_class=1, oversample_ratio=1.0, epochs=epochs, batch_size=batch_size
)
logger.info(f"PyTorch GAN - Class distribution: {pd.Series(y_train_gan).value_counts().to_dict()}")
gan_syn_X, gan_syn_y = extract_synthetic_tail(X_train, X_train_gan, y_train, y_train_gan)

# 5. CTGAN
logger.info("Training CTGAN")
X_train_ctgan, y_train_ctgan, gen_losses_ctgan, disc_losses_ctgan = oversample_with_ctgan(
    X_train, y_train, target_class=1, oversample_ratio=1.0, epochs=epochs, batch_size=batch_size
)
logger.info(f"CTGAN - Class distribution: {pd.Series(y_train_ctgan).value_counts().to_dict()}")
ctgan_syn_X, ctgan_syn_y = extract_synthetic_tail(X_train, X_train_ctgan, y_train, y_train_ctgan)

# 6. Conditional WGAN-GP
logger.info("Training Conditional WGAN-GP")
X_train_cwgangp, y_train_cwgangp, gen_losses_cwgangp, disc_losses_cwgangp = oversample_with_cond_wgangp(
    X_train, y_train, target_class=1, target_ratio=1.0, epochs=epochs, batch_size=batch_size
)
logger.info(f"Conditional WGAN-GP - Class distribution: {pd.Series(y_train_cwgangp).value_counts().to_dict()}")
cwgan_syn_X, cwgan_syn_y = extract_synthetic_tail(X_train, X_train_cwgangp, y_train, y_train_cwgangp)

# === SYNTHETIC DATA EVALUATION ===
logger.info("Evaluating synthetic data quality")
synth_eval_rows = []

def _append_synth_eval(method_name, syn_X, syn_y):
    if syn_X is None or X_test is None or y_test is None:
        return
    
    metrics = evaluate_synthetic_data(
        X_real=X_train,
        X_syn=syn_X,
        X_test=X_test,
        y_test=y_test,
        y_syn=syn_y,
        y_real=y_train,
        seed=random_state,
    )
    metrics["method"] = method_name
    synth_eval_rows.append(metrics)

_append_synth_eval("PyTorch_GAN", gan_syn_X, gan_syn_y)
_append_synth_eval("CTGAN", ctgan_syn_X, ctgan_syn_y)
_append_synth_eval("Conditional_WGAN_GP", cwgan_syn_X, cwgan_syn_y)

if synth_eval_rows:
    synth_eval_df = pd.DataFrame(synth_eval_rows)
    synth_eval_path = os.path.join(
        project_root, "results", f"synth_eval_{DATASET_NAME.replace('.csv', '')}_{epochs}.csv"
    )
    synth_eval_df.to_csv(synth_eval_path, index=False)
    logger.info("Synthetic evaluation saved to %s", synth_eval_path)

# === MODEL TRAINING ===
params = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (len(y_train) / y_train.sum()),
}

training_data = {
    'Baseline XGB': (X_train, y_train),
    'XGB with SMOTE': (X_train_smote, y_train_smote),
    'XGB with Borderline SMOTE': (X_train_blsmote, y_train_blsmote),
    'XGB with SMOTEENN': (X_train_smoteenn, y_train_smoteenn),
    'XGB with PyTorch GAN': (X_train_gan, y_train_gan),
    'XGB with CTGAN': (X_train_ctgan, y_train_ctgan),
    'XGB with Conditional WGAN-GP': (X_train_cwgangp, y_train_cwgangp),
}

logger.info("Training models")
models_dict = {}
for model_name, (X, y) in training_data.items():
    logger.info(f"Training {model_name}")
    model = XGBClassifier(**params, use_label_encoder=False, random_state=random_state)
    model.fit(X, y)
    models_dict[model_name] = model

# === EVALUATION ===
logger.info("Evaluating models")
results_df = evaluate_models_to_dataframe(models_dict, X_test, y_test)

# === SAVE RESULTS ===
results_path = os.path.join(project_root, 'results', f'xgb_{DATASET_NAME.replace(".csv", "")}_{epochs}_results.csv')
results_df.to_csv(results_path, index=False)
logger.info(f"Results saved to {results_path}")
