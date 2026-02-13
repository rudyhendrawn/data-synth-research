import warnings
warnings.filterwarnings("ignore")

import json
import hashlib
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# Utilities
def get_device(prefer: str | None = None) -> str:
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


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    # indices: (B,)
    return torch.nn.functional.one_hot(indices.long(), num_classes=num_classes).float()


# Conditional Generator
class ConditionalGenerator(nn.Module):
    """
    G: [z, cond] -> x_hat
    Output in [-1, 1] because we use tanh, and input is MinMaxScaler(-1,1).
    """
    def __init__(self, noise_dim: int, cond_dim: int, data_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = noise_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=1)
        
        return self.net(x)


# PacGAN Discriminator (Critic)
class PacDiscriminator(nn.Module):
    """
    D: [pack(x), pack(cond)] -> score (WGAN critic)
    If pac = 5, we concatenate 5 samples into 1 "packed" sample.
    This makes it easier to detect lack of diversity / mode collapse.
    """
    def __init__(self, data_dim: int, cond_dim: int, hidden_dim: int = 256, pac: int = 5, dropout: float = 0.0):
        super().__init__()
        self.pac = pac
        in_dim = (data_dim + cond_dim) * pac

        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(hidden_dim, 1),  # critic score, no sigmoid
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x_pack: torch.Tensor, c_pack: torch.Tensor) -> torch.Tensor:
        input = torch.cat([x_pack, c_pack], dim=1)
        
        return self.net(input)


def pack_samples(x: torch.Tensor, c: torch.Tensor, pac: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, data_dim), c: (batch_size, cond_dim)
        Returns packed tensors:
            x_pack: (batch_size//pac, data_dim*pac)
            c_pack: (batch_size//pac, cond_dim*pac)
        Drops remainder if batch_size not divisible by pac.
        """
        batch_size = x.size(0)
        packable_batch = (batch_size // pac) * pac
        if packable_batch == 0:
                raise ValueError("Batch size too small for chosen pac. Increase batch_size or reduce pac.")
        x = x[:packable_batch]
        c = c[:packable_batch]
        x_pack = x.view(packable_batch // pac, -1)
        c_pack = c.view(packable_batch // pac, -1)
        
        return x_pack, c_pack


# Conditional WGAN-GP (cluster-conditioned)
class ConditionalWGAN_GP_Tabular:
    """
    A "CTGAN-like" tabular generator via:
      - cluster-conditioned WGAN-GP
      - PacGAN discriminator
      - MinMaxScaler(-1,1) + tanh output
    """

    def __init__(
        self,
        data_dim: int,
        noise_dim: int = 128,
        hidden_dim: int = 256,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        betas: tuple[float, float] = (0.0, 0.9),
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        pac: int = 5,
        n_clusters: int = 8,
        device: str = "cpu",
        feature_matching: bool = True,
        fm_lambda: float = 1.0,
        seed: int = 42,
    ):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.pac = pac
        self.n_clusters = n_clusters
        self.device = device
        self.feature_matching = feature_matching
        self.fm_lambda = fm_lambda

        # scaler aligned with tanh
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        # KMeans for pseudo-modes
        self._kmeans_seed = seed
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self._kmeans_seed, n_init="auto")

        # Build nets
        self.generator = ConditionalGenerator(noise_dim, n_clusters, data_dim, hidden_dim).to(device)
        self.discriminator = PacDiscriminator(data_dim, n_clusters, hidden_dim, pac=pac, dropout=0.0).to(device)

        # Optimizers (WGAN-GP commonly uses Adam with (0, 0.9))
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas)

        # For feature matching, we expose a feature extractor from discriminator
        # We'll use the first hidden layer output by recreating a small forward hook.
        self._last_features = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        # Hook into first Linear->LeakyReLU block output
        # Assumes self.discriminator.net[0] is Linear and self.discriminator.net[1] is activation
        def hook(_module, _inp, out):
            self._last_features = out

        # attach hook to the first activation layer
        if len(self.discriminator.net) >= 2:
            self.discriminator.net[1].register_forward_hook(hook)

    def _gradient_penalty(self, real_x: torch.Tensor, fake_x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        GP over packed samples.
        We must compute GP on packed inputs consistently.
        """
        total_samples = real_x.size(0)
        # must be divisible by pac; drop remainder
        packable_samples = (total_samples // self.pac) * self.pac
        real_x = real_x[:packable_samples]
        fake_x = fake_x[:packable_samples]
        cond = cond[:packable_samples]

        alpha = torch.rand(packable_samples, 1, device=self.device)
        alpha = alpha.expand_as(real_x)
        interp = alpha * real_x + (1 - alpha) * fake_x
        interp.requires_grad_(True)

        packed_data, packed_condition = pack_samples(interp, cond, self.pac)
        d_interp = self.discriminator(packed_data, packed_condition)

        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grads = grads.view(grads.size(0), -1)
        grads_pen = ((grads.norm(2, dim=1) - 1.0) ** 2).mean() * self.lambda_gp
        
        return grads_pen

    def fit(
        self,
        minority_data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        print_interval: int = 50,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 1e-3,
    ):
        """
        Train on minority-only data.
        Uses cluster conditioning derived from KMeans on scaled minority data.
        """
        # 1) Scale to [-1, 1]
        scaled_minority = self.scaler.fit_transform(minority_data)

        # 2) Fit KMeans on scaled data, get cluster labels
        # If minority is very small, reduce clusters automatically
        sample_count = scaled_minority.shape[0]
        if sample_count < self.n_clusters:
            self.n_clusters = max(2, min(self.n_clusters, sample_count))
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self._kmeans_seed, n_init="auto")
            # rebuild G & D with new cond dim
            self.generator = ConditionalGenerator(self.noise_dim, self.n_clusters, self.data_dim, self.hidden_dim).to(self.device)
            self.discriminator = PacDiscriminator(self.data_dim, self.n_clusters, self.hidden_dim, pac=self.pac, dropout=0.0).to(self.device)
            self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.generator_optimizer.param_groups[0]["lr"], betas=self.generator_optimizer.param_groups[0]["betas"])
            self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_optimizer.param_groups[0]["lr"], betas=self.discriminator_optimizer.param_groups[0]["betas"])
            self._register_feature_hook()

        labels = self.kmeans.fit_predict(scaled_minority)

        X_tensor = torch.tensor(scaled_minority, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)

        # minority-aware batch sizing
        batch_size = int(min(batch_size, len(X_tensor)))
        batch_size = max(32, batch_size) if len(X_tensor) >= 32 else len(X_tensor)

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True, drop_last=False)

        generator_loss_history, discriminator_loss_history = [], []
        def _is_stable() -> bool:
            if not early_stopping:
                return False
            if len(generator_loss_history) < (early_stopping_patience + 1):
                return False
            gen_diffs = [
                abs(generator_loss_history[-i] - generator_loss_history[-i - 1])
                for i in range(1, early_stopping_patience + 1)
            ]
            disc_diffs = [
                abs(discriminator_loss_history[-i] - discriminator_loss_history[-i - 1])
                for i in range(1, early_stopping_patience + 1)
            ]
            return max(gen_diffs) < early_stopping_delta and max(disc_diffs) < early_stopping_delta

        for epoch in range(1, epochs + 1):
            generator_epoch_loss, discriminator_epoch_loss = 0.0, 0.0
            steps = 0

            for real_x, cluster_idx in loader:
                # Ensure we have enough samples for packing
                if real_x.size(0) < self.pac:
                    continue

                cond = one_hot(cluster_idx, self.n_clusters).to(self.device)
                packed_real_data, packed_real_cond = pack_samples(real_x, cond, self.pac)

                # Train Critic (n_critic steps)
                for _ in range(self.n_critic):
                    self.discriminator_optimizer.zero_grad(set_to_none=True)

                    # real
                    real_score = self.discriminator(packed_real_data, packed_real_cond).mean()

                    # fake
                    z = torch.randn(real_x.size(0), self.noise_dim, device=self.device)
                    fake_x = self.generator(z, cond).detach()
                    packed_fake_data, packed_fake_cond = pack_samples(fake_x, cond, self.pac)
                    fake_score = self.discriminator(packed_fake_data, packed_fake_cond).mean()

                    gp = self._gradient_penalty(real_x, fake_x, cond)

                    d_loss = -(real_score - fake_score) + gp
                    d_loss.backward()
                    self.discriminator_optimizer.step()

                    discriminator_epoch_loss += float(d_loss.item())

                # Train Generator
                self.generator_optimizer.zero_grad(set_to_none=True)

                z = torch.randn(real_x.size(0), self.noise_dim, device=self.device)
                generated_samples = self.generator(z, cond)
                packed_generated_data, packed_generated_cond = pack_samples(generated_samples, cond, self.pac)

                # adversarial loss
                g_adv = -self.discriminator(packed_generated_data, packed_generated_cond).mean()

                if self.feature_matching:
                    # Feature matching: match discriminator intermediate features between real and fake
                    _ = self.discriminator(packed_real_data, packed_real_cond)
                    real_feat = self._last_features.detach() if self._last_features is not None else None

                    _ = self.discriminator(packed_generated_data, packed_generated_cond)
                    fake_feat = self._last_features if self._last_features is not None else None

                    if real_feat is not None and fake_feat is not None:
                        fm_loss = (real_feat.mean(dim=0) - fake_feat.mean(dim=0)).abs().mean()
                        g_loss = g_adv + self.fm_lambda * fm_loss
                    else:
                        g_loss = g_adv
                else:
                    g_loss = g_adv

                g_loss.backward()
                self.generator_optimizer.step()

                generator_epoch_loss += float(g_loss.item())
                steps += 1

            if steps == 0:
                # If dataset too small for pac, reduce pac
                raise RuntimeError(
                    f"No training steps executed. Your batch sizes are too small for pac={self.pac}. "
                    f"Reduce pac or increase batch_size."
                )

            generator_loss_history.append(generator_epoch_loss / steps)
            discriminator_loss_history.append(discriminator_epoch_loss / (steps * self.n_critic))

            if epoch % print_interval == 0:
                print(f"[Epoch {epoch:>4}/{epochs}]  G: {generator_loss_history[-1]:.4f}  D: {discriminator_loss_history[-1]:.4f}")
            if _is_stable():
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(Δ< {early_stopping_delta} for {early_stopping_patience} epochs)"
                )
                break

        return generator_loss_history, discriminator_loss_history

    @torch.no_grad()
    def sample(self, n_samples: int, cluster_prior: str = "empirical") -> np.ndarray:
        """
        Generate synthetic samples:
          cluster_prior:
            - 'empirical': sample cluster ids according to training distribution
            - 'uniform': uniform over clusters
        """
        self.generator.eval()

        # pick cluster ids
        if cluster_prior == "uniform":
            cluster_idx = torch.randint(0, self.n_clusters, (n_samples,), device=self.device)
        else:
            # empirical: approximate by sampling from KMeans labels distribution
            # Note: KMeans doesn't store label distribution, so we approximate using cluster sizes via kmeans.labels_
            if hasattr(self.kmeans, "labels_"):
                labels = self.kmeans.labels_
                probs = np.bincount(labels, minlength=self.n_clusters).astype(np.float64)
                probs = probs / probs.sum()
                cluster_np = np.random.choice(np.arange(self.n_clusters), size=n_samples, p=probs)
                cluster_idx = torch.tensor(cluster_np, device=self.device)
            else:
                cluster_idx = torch.randint(0, self.n_clusters, (n_samples,), device=self.device)

        cond = one_hot(cluster_idx, self.n_clusters).to(self.device)
        z = torch.randn(n_samples, self.noise_dim, device=self.device)
        x_syn = self.generator(z, cond).cpu().numpy()

        # inverse scale back to original feature space
        x_inv = self.scaler.inverse_transform(x_syn)
        return x_inv

def oversample_with_cond_wgangp(
    X_train,
    y_train,
    target_class: int = 1,
    target_ratio: float = 1.0,     # 1.0 means match majority size (full balance)
    epochs: int = 100,
    batch_size: int = 128,
    noise_dim: int = 128,
    hidden_dim: int = 256,
    n_critic: int = 5,
    train_max_minority_ratio: float = 1.0,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 1e-3,
    cache_path: str | None = None,
    cache_tag: str | None = None,
    seed: int = 42,
    n_clusters: int = 8,
    pac: int = 5,
    device_prefer: str | None = None,
):
    """
    Oversample minority class using cluster-conditional WGAN-GP + PacGAN.

    target_ratio:
      - If 1.0 -> minority will be grown to match majority size (balanced)
      - If 0.1 -> minority will be grown so that minority ≈ 10% of total

        Returns:
            X_balanced (np.ndarray), y_balanced (np.ndarray), generator_loss_history, discriminator_loss_history
    """
    if isinstance(X_train, pd.DataFrame):
        X_np = X_train.to_numpy()
    else:
        X_np = np.asarray(X_train)

    if isinstance(y_train, pd.Series):
        y_np = y_train.to_numpy(dtype=np.int64)
    else:
        y_np = np.asarray(y_train, dtype=np.int64)

    minority_mask = (y_np == target_class)
    majority_mask = np.logical_not(minority_mask)

    minority_data = X_np[minority_mask]
    n_min = minority_data.shape[0]
    n_maj = int(np.count_nonzero(majority_mask))

    print("Original class distribution:")
    print(f"Majority class (0): {n_maj} samples")
    print(f"Minority class (1): {n_min} samples")

    if target_ratio >= 1.0:
        target_min = n_maj  # balance
    else:
        # make minority about target_ratio of total: min / (maj + min) = r  -> min = r*maj/(1-r)
        target_min = int((target_ratio * n_maj) / max(1e-9, (1 - target_ratio)))

    n_generate = max(0, target_min - n_min)
    if n_generate <= 0:
        print("No oversampling needed.")
        
        return X_np, y_np, [], []

    if 0 < train_max_minority_ratio < 1.0:
        n_keep = max(1, int(n_min * train_max_minority_ratio))
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_min, size=n_keep, replace=False)
        minority_data_train = minority_data[idx]
    else:
        minority_data_train = minority_data

    def _cache_paths():
        if not cache_path:
            return None, None
        os.makedirs(cache_path, exist_ok=True)
        payload = json.dumps(
            {
                "method": "cond_wgan_gp",
                "target_class": target_class,
                "target_ratio": target_ratio,
                "n_generate": n_generate,
                "epochs": epochs,
                "batch_size": batch_size,
                "noise_dim": noise_dim,
                "hidden_dim": hidden_dim,
                "n_critic": n_critic,
                "train_max_minority_ratio": train_max_minority_ratio,
                "n_clusters": n_clusters,
                "pac": pac,
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()[:10]
        tag = cache_tag or "dataset"
        base = f"{tag}_cond_wgan_gp_{digest}"
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
            X_bal = np.vstack((X_np, cached_x))
            y_bal = np.concatenate((np.asarray(y_np, dtype=np.int64), cached_y))
            return X_bal, y_bal, [], []

    device = get_device(prefer=device_prefer)
    print(f"Using device: {device}")

    model = ConditionalWGAN_GP_Tabular(
        data_dim=minority_data.shape[1],
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        lr_g=2e-4,
        lr_d=2e-4,
        betas=(0.0, 0.9),
        n_critic=n_critic,
        lambda_gp=10.0,
        pac=pac,
        n_clusters=n_clusters,
        device=device,
        feature_matching=True,
        fm_lambda=1.0,
        seed=seed,
    )

    # minority-aware batch sizing (very important in your setup)
    bs = min(batch_size, n_min)
    bs = max(32, bs) if n_min >= 32 else n_min

    print(
        f"Training Conditional WGAN-GP (clusters={n_clusters}, pac={pac}, batch_size={bs}) "
        f"for {epochs} epochs (n_critic={n_critic}, early_stopping={early_stopping})..."
    )
    generator_loss_history, discriminator_loss_history = model.fit(
        minority_data_train,
        epochs=epochs,
        batch_size=bs,
        print_interval=50,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
    )

    print(f"Generating {n_generate} synthetic samples...")
    synthetic = model.sample(n_generate, cluster_prior="empirical")
    synth_labels = np.full(n_generate, target_class, dtype=np.int64)
    if cache_x_path and cache_y_path:
        np.save(cache_x_path, synthetic)
        np.save(cache_y_path, synth_labels)
        print(f"Saved synthetic samples to {cache_x_path}")

    X_bal = np.vstack((X_np, synthetic))
    y_bal = np.concatenate((np.asarray(y_np, dtype=np.int64), synth_labels))

    print("\nFinal class distribution after Conditional WGAN-GP oversampling:")
    print(f"Majority class (0): {int((y_bal == 0).sum())} samples")
    print(f"Minority class (1): {int((y_bal == 1).sum())} samples")

    return X_bal, y_bal, generator_loss_history, discriminator_loss_history
