import torch
import numpy as np

from model import TransformerDenoiser1D, GaussianDiffusion1D


@torch.no_grad()
def generate_phrase(
    checkpoint_path,
    device="cpu",
    seq_len=None,
    n_samples=1
):
    device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    params = checkpoint["model_params"]
    if seq_len is not None and seq_len != params["seq_len"]:
        raise ValueError(
            f"Checkpoint expects seq_len={params['seq_len']}, "
            f"but received seq_len={seq_len}"
        )

    model = TransformerDenoiser1D(**params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = GaussianDiffusion1D(
        timesteps=checkpoint["timesteps"],
        beta_start=checkpoint.get("beta_start", 1e-4),
        beta_end=checkpoint.get("beta_end", 0.02),
        device=device
    )

    x = torch.randn(
        n_samples, 1, params["seq_len"], device=device
    )

    for step in reversed(range(diffusion.timesteps)):
        t = torch.full(
            (n_samples,), step, device=device, dtype=torch.long
        )
        beta = diffusion.betas[step]
        alpha = diffusion.alphas[step]
        alpha_bar = diffusion.alpha_bars[step]

        predicted_noise = model(x, t)
        x = (
            x - (beta / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
        ) / torch.sqrt(alpha)

        if step > 0:
            alpha_bar_prev = diffusion.alpha_bars[step - 1]
            variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            x = x + torch.sqrt(variance.clamp_min(1e-20)) * torch.randn_like(x)

    generated = x.squeeze(1).cpu().numpy()

    mean = float(checkpoint["contour_mean"])
    std = float(checkpoint["contour_std"])
    generated = generated * std + mean

    if n_samples == 1:
        return generated[0]
    return generated
