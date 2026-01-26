from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, probs, y_true, max_iter: int = 1000, lr: float = 0.01) -> float:
        y_arr = np.asarray(y_true, dtype=np.float32)
        probs_arr = np.asarray(probs, dtype=np.float32)
        probs_arr = np.clip(probs_arr, 1e-6, 1 - 1e-6)
        logits = np.log(probs_arr / (1 - probs_arr)).astype(np.float32)

        logits_t = torch.tensor(logits)
        y_t = torch.tensor(y_arr)

        temp = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([temp], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = logits_t / temp.clamp(min=1e-3)
            loss = loss_fn(scaled_logits, y_t)
            loss.backward()
            optimizer.step()

        self.temperature = float(temp.clamp(min=1e-3).detach().cpu().item())
        return self.temperature

    def transform(self, probs):
        probs_arr = np.asarray(probs, dtype=np.float32)
        probs_arr = np.clip(probs_arr, 1e-6, 1 - 1e-6)
        logits = np.log(probs_arr / (1 - probs_arr))
        scaled = logits / max(self.temperature, 1e-6)
        return 1.0 / (1.0 + np.exp(-scaled))
