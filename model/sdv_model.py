import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer

def oversample_with_sdv_ctgan(X_train, y_train, target_class=1, oversample_ratio=1.0, batch_size=512, epochs=500):
    """
    Oversample minority class using SDV's CTGANSynthesizer.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        target_class (int): The minority class label to oversample.
        oversample_ratio (float): The ratio of oversampling.
        epochs (int): Number of training epochs for CTGAN.
    
    Returns:
        X_balanced (pd.DataFrame): Balanced training features.
        y_balanced (pd.Series): Balanced training labels.
    """
    # Convert to numpy arrays if needed
    X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
    y_np = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    
    minority_mask = y_np == target_class
    majority_mask = y_np != target_class
    
    minority_data = X_np[minority_mask]
    n_majority = np.sum(majority_mask)
    n_minority = len(minority_data)
    
    print("Original class distribution:")
    print(f"Majority class (0): {n_majority} samples")
    print(f"Minority class (1): {n_minority} samples")
    
    # Calculate number of samples to generate
    n_generate = int(n_majority * oversample_ratio) - n_minority
    
    if n_generate <= 0:
        print("No oversampling needed")
        return X_train, y_train
    
    # Prepare minority data as DataFrame
    if isinstance(X_train, pd.DataFrame):
        minority_df = X_train[minority_mask].copy()
    else:
        minority_df = pd.DataFrame(minority_data, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(minority_df)
    
    # Initialize CTGAN
    print(f"Training SDV CTGAN for {epochs} epochs...")
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        verbose=True
    )
    
    # Fit CTGAN on minority class
    synthesizer.fit(minority_df)
    
    # Generate synthetic samples
    print(f"Generating {n_generate} synthetic samples...")
    synthetic_frames = []
    chunk_size = 20000
    for start in range(0, n_generate, chunk_size):
        end = min(start + chunk_size, n_generate)
        synthetic_frames.append(synthesizer.sample(num_rows=end - start))
    synthetic_df = pd.concat(synthetic_frames, ignore_index=True)
    # synthetic_df = synthesizer.sample(num_rows=n_generate)
    
    # Combine original and synthetic data
    if isinstance(X_train, pd.DataFrame):
        X_balanced = pd.concat([X_train, synthetic_df], ignore_index=True)
        y_balanced = pd.concat([
            y_train,
            pd.Series([target_class] * n_generate)
        ], ignore_index=True)
    else:
        X_balanced = np.vstack([X_np, synthetic_df.values])
        y_balanced = np.concatenate([np.asarray(y_np), np.full(n_generate, target_class)])
        # y_balanced = np.concatenate([np.asarray(y_np), synthetic_labels])
        
    
    print(f"\nFinal class distribution after SDV CTGAN oversampling:")
    print(f"Majority class (0): {np.sum(y_balanced == 0)} samples")
    print(f"Minority class (1): {np.sum(y_balanced == 1)} samples")
    
    return X_balanced, y_balanced


def oversample_with_gaussian_copula(X_train, y_train, target_class=1, oversample_ratio=1.0):
    """
    Oversample minority class using SDV's GaussianCopulaSynthesizer (faster alternative).
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        target_class (int): The minority class label to oversample.
        oversample_ratio (float): The ratio of oversampling.
    
    Returns:
        X_balanced (pd.DataFrame): Balanced training features.
        y_balanced (pd.Series): Balanced training labels.
    """
    # Convert to numpy arrays if needed
    X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
    y_np = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    
    minority_mask = y_np == target_class
    majority_mask = y_np != target_class
    
    minority_data = X_np[minority_mask]
    n_majority = np.sum(majority_mask)
    n_minority = len(minority_data)
    
    print("Original class distribution:")
    print(f"Majority class (0): {n_majority} samples")
    print(f"Minority class (1): {n_minority} samples")
    
    # Calculate number of samples to generate
    n_generate = int(n_majority * oversample_ratio) - n_minority
    
    if n_generate <= 0:
        print("No oversampling needed")
        return X_train, y_train
    
    # Prepare minority data as DataFrame
    if isinstance(X_train, pd.DataFrame):
        minority_df = X_train[minority_mask].copy()
    else:
        minority_df = pd.DataFrame(minority_data, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(minority_df)
    
    # Initialize Gaussian Copula (faster than CTGAN)
    print(f"Training Gaussian Copula synthesizer...")
    synthesizer = GaussianCopulaSynthesizer(
        metadata=metadata,
        enforce_min_max_values=True,
        enforce_rounding=False
    )
    
    # Fit synthesizer on minority class
    synthesizer.fit(minority_df)
    
    # Generate synthetic samples
    print(f"Generating {n_generate} synthetic samples...")
    synthetic_df = synthesizer.sample(num_rows=n_generate)
    
    # Combine original and synthetic data
    if isinstance(X_train, pd.DataFrame):
        X_balanced = pd.concat([X_train, synthetic_df], ignore_index=True)
        y_balanced = pd.concat([
            y_train,
            pd.Series([target_class] * n_generate)
        ], ignore_index=True)
    else:
        X_balanced = np.vstack([X_np, synthetic_df.values])
        y_balanced = np.concatenate([np.asarray(y_np), np.full(n_generate, target_class)])
    
    print(f"\nFinal class distribution after Gaussian Copula oversampling:")
    print(f"Majority class (0): {np.sum(y_balanced == 0)} samples")
    print(f"Minority class (1): {np.sum(y_balanced == 1)} samples")
    
    return X_balanced, y_balanced

def oversample_with_copula_gan(X_train, y_train, target_class=1, oversample_ratio=1.0, batch_size=512, epochs=500):
    """
    Oversample minority class using SDV's CopulaGAN synthesizer.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        target_class (int): The minority class label to oversample.
        oversample_ratio (float): The ratio of oversampling.
        epochs (int): Number of training epochs for CopulaGAN.
    
    Returns:
        X_balanced (pd.DataFrame): Balanced training features.
        y_balanced (pd.Series): Balanced training labels.
    """
    # Implementation would be similar to oversample_with_sdv_ctgan,
    # but using CopulaGANSynthesizer instead.
    # Convert to numpy arrays if needed
    X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
    y_np = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
    
    minority_mask = y_np == target_class
    majority_mask = y_np != target_class
    
    minority_data = X_np[minority_mask]
    n_majority = np.sum(majority_mask)
    n_minority = len(minority_data)
    
    print("Original class distribution:")
    print(f"Majority class (0): {n_majority} samples")
    print(f"Minority class (1): {n_minority} samples")
    
    # Calculate number of samples to generate
    n_generate = int(n_majority * oversample_ratio) - n_minority
    
    if n_generate <= 0:
        print("No oversampling needed")
        return X_train, y_train
    
    # Prepare minority data as DataFrame
    if isinstance(X_train, pd.DataFrame):
        minority_df = X_train[minority_mask].copy()
    else:
        minority_df = pd.DataFrame(minority_data, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(minority_df)
    
    # Initialize CopulaGAN
    print(f"Training SDV Copula GAN for {epochs} epochs...")
    synthesizer = CopulaGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # Fit CopulaGAN on minority class
    synthesizer.fit(minority_df)
    
    # Generate synthetic samples
    print(f"Generating {n_generate} synthetic samples...")
    synthetic_frames = []
    chunk_size = 20000
    for start in range(0, n_generate, chunk_size):
        end = min(start + chunk_size, n_generate)
        synthetic_frames.append(synthesizer.sample(num_rows=end - start))
    synthetic_df = pd.concat(synthetic_frames, ignore_index=True)
    # synthetic_df = synthesizer.sample(num_rows=n_generate)
    
    # Combine original and synthetic data
    if isinstance(X_train, pd.DataFrame):
        X_balanced = pd.concat([X_train, synthetic_df], ignore_index=True)
        y_balanced = pd.concat([
            y_train,
            pd.Series([target_class] * n_generate)
        ], ignore_index=True)
    else:
        X_balanced = np.vstack([X_np, synthetic_df.values])
        y_balanced = np.concatenate([np.asarray(y_np), np.full(n_generate, target_class)])
    
    print(f"\nFinal class distribution after SDV Copula GAN oversampling:")
    print(f"Majority class (0): {np.sum(y_balanced == 0)} samples")
    print(f"Minority class (1): {np.sum(y_balanced == 1)} samples")
    
    return X_balanced, y_balanced
    