
import torch

from model import TransformerVAE


@torch.no_grad()
def generate_phrase(
    checkpoint_path,
    device
):
    # Load trained checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Recreate the trained architecture
    model = TransformerVAE(
        input_dim=1,
        seq_len=checkpoint["seq_len"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        latent_dim=checkpoint["latent_dim"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Sample a random latent vector
    z = torch.randn(
        1,
        checkpoint["latent_dim"],
        device=device
    )

    # Generate a pitch contour
    generated = model.decode(z)

    # Convert [1, 128, 1] to [128]
    generated = generated.squeeze().cpu().numpy()

    # Reverse normalization
    generated = (
        generated * checkpoint["contour_std"]
        + checkpoint["contour_mean"]
    )

    return generated
