
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model import TransformerVAE, vae_loss


def train_model(
    dataloader,
    dataset,
    contour_mean,
    contour_std,
    device,
    output_dir,
    epochs=300,
    learning_rate=0.001,
    seq_len=128,
    d_model=64,
    nhead=4,
    num_layers=2,
    latent_dim=2
):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Transformer-VAE
    model = TransformerVAE(
        input_dim=1,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        latent_dim=latent_dim
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_history = []

    print("Transformer-VAE training started")
    print("Device:", device)
    print("Training instances:", len(dataset))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x in dataloader:
            x = x.to(device)

            # Convert [batch, 1, 128] to [batch, 128, 1]
            x = x.transpose(1, 2)

            optimizer.zero_grad()

            reconstructed, mu, logvar = model(x)

            loss = vae_loss(
                x, reconstructed, mu, logvar
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += x.size(0)

        epoch_loss = total_loss / total_samples
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"| Loss: {epoch_loss:.4f}"
            )

    # Save checkpoint and configuration
    checkpoint_path = os.path.join(
        output_dir,
        "transformer_vae_model.pt"
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "seq_len": seq_len,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "latent_dim": latent_dim,
        "contour_mean": float(contour_mean),
        "contour_std": float(contour_std),
        "loss_history": loss_history
    }, checkpoint_path)

    # Save loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss per Sample")
    plt.title("Transformer-VAE Training Loss")
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "training_loss.png")
    )
    plt.show()
    plt.close()

    print("Transformer-VAE training completed!")
    print("Checkpoint saved:", checkpoint_path)

    return checkpoint_path, loss_history
