
import torch

from model import VAE


@torch.no_grad()
def generate_phrase(
    checkpoint_path,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = VAE(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Sample a random latent vector
    z = torch.randn(
        1,
        checkpoint["latent_dim"],
        device=device
    )

    # Decode into a pitch contour
    generated = model.decode(z)

    generated = generated.squeeze(0).cpu().numpy()

    # Reverse normalization
    generated = (
        generated * checkpoint["contour_std"]
        + checkpoint["contour_mean"]
    )

    return generated
