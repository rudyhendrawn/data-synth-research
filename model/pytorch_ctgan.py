import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

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
    def __init__(self, data_dim, noise_dim=100, hidden_dim=128, lr=2e-4, device='mps'):
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
    
    def train(self, minority_data, epochs=500, batch_size=64, print_interval=100, n_critic=5):
        # Scale data
        minority_data_scaled = self.scaler.fit_transform(minority_data)
        minority_tensor = torch.FloatTensor(minority_data_scaled).to(self.device)
        dataloader = DataLoader(TensorDataset(minority_tensor), batch_size=batch_size, shuffle=True)

        gen_losses = []
        disc_losses = []

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
def oversample_with_ctgan(X_train, y_train, target_class=1, oversample_ratio=1.0, epochs=500, batch_size=64):
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
    
    # Setup device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize and train CTGAN oversampler
    ctgan_oversampler = SimplifiedCTGAN(
        data_dim=minority_data.shape[1],
        noise_dim=100,
        hidden_dim=128,
        lr=0.0002,
        device=device
    )

    print(f"Training Simplified CTGAN for {epochs} epochs...")
    gen_losses, disc_losses = ctgan_oversampler.train(
        minority_data, 
        epochs=epochs, 
        batch_size=batch_size, 
        print_interval=100
    )

    # Generate synthetic samples
    print(f"Generating {n_generate} synthetic samples...")
    synthetic_samples = ctgan_oversampler.generate_samples(n_generate)
    synthetic_labels = np.full(n_generate, target_class, dtype=np.int64)

    # Combine data
    X_balanced = np.vstack([X_np, synthetic_samples])
    y_balanced = np.concatenate([np.asarray(y_np), synthetic_labels])

    print(f"\nFinal class distribution after CTGAN oversampling:")
    print(f"Majority class (0): {np.sum(y_balanced == 0)} samples")
    print(f"Minority class (1): {np.sum(y_balanced == 1)} samples")

    return X_balanced, y_balanced, gen_losses, disc_losses