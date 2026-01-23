# Implement GAN and CTGAN for data synthesis
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


# Define PyTorch GAN components
class Generator(nn.Module):
    def __init__(self, noise_dim, data_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # 128 -> 256
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),  # 256 -> 128
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),    # 128 -> data_dim
            # nn.Tanh()
        )

    def forward(self, noise):
        return self.network(noise)

class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.network(data)

class GANOversampler:
    def __init__(self, data_dim, noise_dim=100, hidden_dim=128, lr=0.0002, device='mps'):
        self.device = device
        self.noise_dim = noise_dim
        self.data_dim = data_dim

        self.generator = Generator(noise_dim, data_dim, hidden_dim).to(device)
        self.discriminator = Discriminator(data_dim, hidden_dim).to(device)

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()
        self.scaler = StandardScaler()

    def train(self, minority_data, epochs=500, batch_size=64, print_interval=100):
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

                # Train Discriminator
                self.disc_optimizer.zero_grad()

                real_labels = torch.ones(current_batch_size, 1).to(self.device)
                real_output = self.discriminator(real_data)
                real_loss = self.criterion(real_output, real_labels)

                noise = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                fake_loss = self.criterion(fake_output, fake_labels)

                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                self.disc_optimizer.step()

                # Train Generator
                self.gen_optimizer.zero_grad()
                noise = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data)
                gen_loss = self.criterion(fake_output, real_labels)

                gen_loss.backward()
                self.gen_optimizer.step()

                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item()
            
            gen_losses.append(epoch_gen_loss / len(dataloader))
            disc_losses.append(epoch_disc_loss / len(dataloader))

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

        
# Pytorch GAN-based oversampling function
def oversample_with_pytorch_gan(X_train, y_train, target_class=1, oversample_ratio=1.0, epochs=500, batch_size=64):
    """
    Oversample minority class using Pytorch GAN.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        target_class (int): The minority class label to oversample.
        oversample_ratio (float): The ratio of oversampling. E.g., 1.0 means double the minority class.
        epochs (int): Number of training epochs for GAN.
        batch_size (int): Batch size for GAN training.
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

    # Initialize and train GAN oversampler
    gan_oversampler = GANOversampler(
        data_dim=minority_data.shape[1],
        noise_dim=100,
        hidden_dim=128,
        lr=0.0002,
        device=device
    )

    print(f"Training PyTorch GAN for {epochs} epochs...")
    gen_losses, disc_losses = gan_oversampler.train(
        minority_data, 
        epochs=epochs, 
        batch_size=batch_size, 
        print_interval=100
    )
    
    # Generate synthetic samples
    print(f"Generating {n_generate} synthetic samples...")
    synthetic_data = gan_oversampler.generate_samples(n_generate)
    synthetic_labels = np.full(n_generate, target_class)

    # Combine original and synthetic data
    X_balanced = np.vstack([X_np, synthetic_data])
    y_balanced = np.concatenate([np.asarray(y_np), synthetic_labels])

    print(f"\nFinal class distribution after Pytorch GAN oversampling:")
    print(f"Majority class (0): {np.sum(y_balanced == 0)} samples")
    print(f"Minority class (1): {np.sum(y_balanced == 1)} samples")

    return X_balanced, y_balanced, gen_losses, disc_losses
