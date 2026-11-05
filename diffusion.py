
import torch


class GaussianDiffusion1D:
    def __init__(self, timesteps=500, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(
            1e-4, 0.02, timesteps, device=device
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
