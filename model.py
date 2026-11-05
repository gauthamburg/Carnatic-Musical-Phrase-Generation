
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2

        scale = torch.exp(
            -np.log(10000) *
            torch.arange(half, device=t.device).float() /
            max(half - 1, 1)
        )

        angles = t.float().unsqueeze(1) * scale.unsqueeze(0)

        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ConditionalUNet1D(nn.Module):
    def __init__(self, num_classes=4, channels=64, time_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.class_embedding = nn.Embedding(num_classes, time_dim)

        self.input_conv = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.down_conv = nn.Conv1d(channels, channels * 2, 4, stride=2, padding=1)
        self.mid_conv = nn.Conv1d(channels * 2, channels * 2, 3, padding=1)
        self.up_conv = nn.ConvTranspose1d(channels * 2, channels, 4, stride=2, padding=1)
        self.output_conv = nn.Conv1d(channels * 2, 1, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, channels * 2)

    def forward(self, x, t, labels):
        t_emb = self.time_embedding(t)
        c_emb = self.class_embedding(labels)

        emb = self.time_proj(t_emb + c_emb).unsqueeze(-1)

        x1 = F.silu(self.input_conv(x))
        x2 = F.silu(self.down_conv(x1))
        x2 = F.silu(self.mid_conv(x2) + emb)

        x3 = F.silu(self.up_conv(x2))
        x = torch.cat([x3, x1], dim=1)

        return self.output_conv(x)



class GaussianDiffusion1D:
    def __init__(self, timesteps=500, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(
            1e-4, 0.02, timesteps, device=self.device
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)

        noisy = (
            torch.sqrt(alpha_bar) * x0 +
            torch.sqrt(1 - alpha_bar) * noise
        )

        return noisy, noise





