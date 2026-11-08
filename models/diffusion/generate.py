
import numpy as np
import torch

from model import ConditionalUNet1D, GaussianDiffusion1D


@torch.no_grad()
def generate_phrase(
    checkpoint_path,
    device,
    seq_len=128
):
    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    timesteps = checkpoint["timesteps"]

    # Recreate the trained architecture
    network = ConditionalUNet1D(
        num_classes=1,
        channels=checkpoint["channels"],
        time_dim=checkpoint["time_dim"]
    ).to(device)

    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()

    diffusion = GaussianDiffusion1D(
        timesteps=timesteps,
        device=device
    )

    # Start with random noise: one phrase
    x = torch.randn(1, 1, seq_len, device=device)

    labels = torch.zeros(
        1,
        dtype=torch.long,
        device=device
    )

    # Reverse diffusion
    for step in reversed(range(timesteps)):

        t = torch.full(
            (1,),
            step,
            dtype=torch.long,
            device=device
        )

        beta = diffusion.betas[step]
        alpha = diffusion.alphas[step]
        alpha_bar = diffusion.alpha_bars[step]

        predicted_noise = network(x, t, labels)

        x = (
            x - beta / torch.sqrt(1 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha)

        if step > 0:
            alpha_bar_prev = diffusion.alpha_bars[step - 1]

            variance = (
                beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
            )

            x += torch.sqrt(variance) * torch.randn_like(x)

    # Convert normalized contour back to cents
    generated = x.squeeze().cpu().numpy()

    generated = (
        generated * checkpoint["contour_std"]
        + checkpoint["contour_mean"]
    )

    return generated
