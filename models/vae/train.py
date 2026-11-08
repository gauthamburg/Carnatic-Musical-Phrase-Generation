
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model import VAE, vae_loss


def train_model(
    dataloader,
    dataset,
    contour_mean,
    contour_std,
    device,
    epochs=300,
    learning_rate=0.001,
    latent_dim=2,
    output_dir="models/vae/checkpoints"
):
    os.makedirs(output_dir, exist_ok=True)

    model = VAE(
        input_dim=128,
        latent_dim=latent_dim
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_history = []

    print("VAE training started")
    print("Device:", device)
    print("Training instances:", len(dataset))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x in dataloader:
            x = x.to(device)
            x = x.reshape(x.size(0), -1)

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

    checkpoint_path = os.path.join(
        output_dir, "vae_model.pt"
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": latent_dim,
        "input_dim": 128,
        "contour_mean": float(contour_mean),
        "contour_std": float(contour_std),
        "loss_history": loss_history
    }, checkpoint_path)

    loss_path = os.path.join(
        output_dir, "training_loss.png"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss per Sample")
    plt.title("VAE Training Loss")
    plt.grid(True)
    plt.savefig(loss_path)
    plt.show()
    plt.close()

    print("VAE training completed!")
    print("Model saved:", checkpoint_path)

    return checkpoint_path, loss_history
