import json
import hashlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

def resolve_device(prefer: str | None = None) -> str:
    order = []
    if prefer in {"cuda", "mps", "cpu"}:
        order.append(prefer)
    for candidate in ("cuda", "mps", "cpu"):
        if candidate not in order:
            order.append(candidate)
    for candidate in order:
        if candidate == "cuda" and torch.cuda.is_available():
            return "cuda"
        if candidate == "mps" and torch.backends.mps.is_available():
            return "mps"
        if candidate == "cpu":
            return "cpu"
    return "cpu"

# CTGAN-based oversampling function using Pytorch CTGAN
class CTGANGenerator(nn.Module):
    def __init__(self, noise_dim, data_dim, hidden_dim=128):  # FIX: swap order
        super(CTGANGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),  # Now correct: 100 -> 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),  # 128 -> 29
            # nn.Tanh()
        )

    def forward(self, noise):
        return self.network(noise)
    
class CTGANDiscriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim=128):
        super(CTGANDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, data):
        return self.network(data)
    
class SimplifiedCTGAN:
    def __init__(self, data_dim, noise_dim=100, hidden_dim=128, lr=2e-4, device='cpu'):
        self.device = device
        self.noise_dim = noise_dim
        self.data_dim = data_dim

        self.generator = CTGANGenerator(noise_dim, data_dim, hidden_dim).to(device)
        self.discriminator = CTGANDiscriminator(data_dim, hidden_dim).to(device)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        self.scaler = StandardScaler()

    def compute_gradient_penalty(self, real_data, fake_data, lambda_gp=10):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        
        return gradient_penalty
    
    def train(
        self,
        minority_data,
        epochs=500,
        batch_size=64,
        print_interval=100,
        n_critic=5,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 1e-3,
    ):
        # Scale data
        minority_data_scaled = self.scaler.fit_transform(minority_data)
        minority_tensor = torch.FloatTensor(minority_data_scaled).to(self.device)
        dataloader = DataLoader(TensorDataset(minority_tensor), batch_size=batch_size, shuffle=True)

        gen_losses = []
        disc_losses = []

        def _is_stable() -> bool:
            if not early_stopping:
                return False
            if len(gen_losses) < (early_stopping_patience + 1):
                return False
            gen_diffs = [
                abs(gen_losses[-i] - gen_losses[-i - 1])
                for i in range(1, early_stopping_patience + 1)
            ]
            disc_diffs = [
                abs(disc_losses[-i] - disc_losses[-i - 1])
                for i in range(1, early_stopping_patience + 1)
            ]
            return max(gen_diffs) < early_stopping_delta and max(disc_diffs) < early_stopping_delta

        for epoch in range(epochs):
            epoch_gen_loss = 0.0
            epoch_disc_loss = 0.0

            for batch_idx, (real_data,) in enumerate(dataloader):
                current_batch_size = real_data.size(0)

                # Train discriminator
                for _ in range(n_critic):
                    self.disc_optimizer.zero_grad()

                    # Real data
                    real_validity = self.discriminator(real_data)

                    noise = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                    fake_data = self.generator(noise).detach()
                    fake_validity = self.discriminator(fake_data)

                    # Gradient penalty
                    gp = self.compute_gradient_penalty(real_data, fake_data)

                    # Wasserstein loss with gradient penalty
                    disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp
                    disc_loss.backward()
                    self.disc_optimizer.step()

                    epoch_disc_loss += disc_loss.item()

                # train generator
                self.gen_optimizer.zero_grad()

                noise = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_validity = self.discriminator(fake_data)
                gen_loss = -torch.mean(fake_validity)
                gen_loss.backward()
                self.gen_optimizer.step()

                epoch_gen_loss += gen_loss.item()
            
            # Store average losses
            gen_losses.append(epoch_gen_loss / len(dataloader))
            disc_losses.append(epoch_disc_loss / (len(dataloader) * n_critic))

            # Print progress every print_interval epochs
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Gen Loss: {gen_losses[-1]:.4f}, Disc Loss: {disc_losses[-1]:.4f}")
            if _is_stable():
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(Δ< {early_stopping_delta} for {early_stopping_patience} epochs)"
                )
                break

        return gen_losses, disc_losses
    
    def generate_samples(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim).to(self.device)
            synthetic_data = self.generator(noise)

        # Inverse transform the data to original scale
        synthetic_data_np = synthetic_data.cpu().numpy()
        synthetic_data_inv = self.scaler.inverse_transform(synthetic_data_np)

        return synthetic_data_inv
    
# Simplified CTGAN-based oversampling function
def oversample_with_ctgan(
    X_train,
    y_train,
    target_class=1,
    oversample_ratio=1.0,
    epochs=100,
    batch_size=128,
    noise_dim: int = 100,
    hidden_dim: int = 128,
    lr: float = 0.0002,
    n_critic: int = 5,
    train_max_minority_ratio: float = 1.0,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 1e-3,
    cache_path: str | None = None,
    cache_tag: str | None = None,
    seed: int = 42,
    device_prefer: str | None = None,
):
    """
    Oversample minority class using simplified CTGAN.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        target_class (int): The minority class label to oversample.
        oversample_ratio (float): The ratio of oversampling. E.g., 1.0 means double the minority class.
        epochs (int): Number of training epochs for CTGAN.
        batch_size (int): Batch size for CTGAN training.
    Returns:
        X_balanced (pd.DataFrame): Balanced training features.
        y_balanced (pd.Series): Balanced training labels.
        gen_losses (list): Generator losses during training.
        disc_losses (list): Discriminator losses during training.
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
        return X_train, y_train, [], []

    if 0 < train_max_minority_ratio < 1.0:
        n_keep = max(1, int(n_minority * train_max_minority_ratio))
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_minority, size=n_keep, replace=False)
        minority_data_train = minority_data[idx]
    else:
        minority_data_train = minority_data

    def _cache_paths():
        if not cache_path:
            return None, None
        os.makedirs(cache_path, exist_ok=True)
        payload = json.dumps(
            {
                "method": "ctgan",
                "target_class": target_class,
                "oversample_ratio": oversample_ratio,
                "n_generate": n_generate,
                "epochs": epochs,
                "batch_size": batch_size,
                "noise_dim": noise_dim,
                "hidden_dim": hidden_dim,
                "n_critic": n_critic,
                "train_max_minority_ratio": train_max_minority_ratio,
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()[:10]
        tag = cache_tag or "dataset"
        base = f"{tag}_ctgan_{digest}"
        return (
            os.path.join(cache_path, f"{base}_X.npy"),
            os.path.join(cache_path, f"{base}_y.npy"),
        )

    cache_x_path, cache_y_path = _cache_paths()
    if cache_x_path and cache_y_path and os.path.exists(cache_x_path) and os.path.exists(cache_y_path):
        cached_x = np.load(cache_x_path)
        cached_y = np.load(cache_y_path)
        if len(cached_x) == n_generate:
            print(f"Loaded cached synthetic samples from {cache_x_path}")
            X_balanced = np.vstack([X_np, cached_x])
            y_balanced = np.concatenate([np.asarray(y_np), cached_y])
            return X_balanced, y_balanced, [], []
    
    # Setup device
    device = resolve_device(device_prefer)
    print(f"Using device: {device}")

    # Initialize and train CTGAN oversampler
    ctgan_oversampler = SimplifiedCTGAN(
        data_dim=minority_data.shape[1],
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device
    )

    print(
        f"Training Simplified CTGAN for {epochs} epochs "
        f"(n_critic={n_critic}, early_stopping={early_stopping}, batch_size={batch_size})..."
    )
    gen_losses, disc_losses = ctgan_oversampler.train(
        minority_data_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        print_interval=100,
        n_critic=n_critic,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
    )

    # Generate synthetic samples
    print(f"Generating {n_generate} synthetic samples...")
    synthetic_samples = ctgan_oversampler.generate_samples(n_generate)
    synthetic_labels = np.full(n_generate, target_class, dtype=np.int64)
    if cache_x_path and cache_y_path:
        np.save(cache_x_path, synthetic_samples)
        np.save(cache_y_path, synthetic_labels)
        print(f"Saved synthetic samples to {cache_x_path}")

    # Combine data
    X_balanced = np.vstack([X_np, synthetic_samples])
    y_balanced = np.concatenate([np.asarray(y_np), synthetic_labels])

    print(f"\nFinal class distribution after CTGAN oversampling:")
    print(f"Majority class (0): {np.sum(y_balanced == 0)} samples")
    print(f"Minority class (1): {np.sum(y_balanced == 1)} samples")

    return X_balanced, y_balanced, gen_losses, disc_losses
