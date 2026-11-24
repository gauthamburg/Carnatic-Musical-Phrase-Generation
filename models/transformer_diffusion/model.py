import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        scale = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=device, dtype=torch.float32) /
            max(half - 1, 1)
        )
        angles = t.float().unsqueeze(1) * scale.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class TransformerDenoiser1D(nn.Module):
    """
    Transformer noise predictor for 1D pitch contours.
    Input/output shape: [batch, 1, seq_len].
    """
    def __init__(
        self,
        seq_len=128,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        time_dim=128
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.seq_len = seq_len
        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x, t):
        # [B, 1, L] -> [B, L, 1]
        x = x.transpose(1, 2)
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.size(1)}"
            )

        h = self.input_projection(x)
        t_emb = self.time_embedding(t).unsqueeze(1)
        h = h + self.position_embedding + t_emb
        h = self.transformer(h)
        h = self.output_projection(self.output_norm(h))

        # [B, L, 1] -> [B, 1, L]
        return h.transpose(1, 2)


class GaussianDiffusion1D:
    """DDPM forward process and coefficients."""
    def __init__(self, timesteps=500, beta_start=1e-4,
                 beta_end=0.02, device="cpu"):
        self.timesteps = int(timesteps)
        self.device = torch.device(device)

        self.betas = torch.linspace(
            beta_start, beta_end, self.timesteps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        xt = (
            torch.sqrt(alpha_bar) * x0 +
            torch.sqrt(1.0 - alpha_bar) * noise
        )
        return xt, noise
