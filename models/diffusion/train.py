
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import ConditionalUNet1D, GaussianDiffusion1D


def train_model(
    dataloader,
    dataset,
    contour_mean,
    contour_std,
    output_dir,
    device,
    epochs=100,
    learning_rate=2e-4,
    timesteps=500,
    channels=64,
    time_dim=128
):
    os.makedirs(output_dir, exist_ok=True)

    # Only one phrase-raga combination is being trained.
    network = ConditionalUNet1D(
        num_classes=1,
        channels=channels,
        time_dim=time_dim
    ).to(device)

    diffusion = GaussianDiffusion1D(
        timesteps=timesteps,
        device=device
    )

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=learning_rate
    )

    loss_history = []

    print("Training started")
    print("Device:", device)
    print("Training instances:", len(dataset))

    for epoch in range(epochs):
        network.train()
        epoch_loss = 0.0

        for x0 in dataloader:
            x0 = x0.to(device)

            # Dummy label: only one class is used.
            labels = torch.zeros(
                x0.size(0),
                dtype=torch.long,
                device=device
            )

            # Random diffusion timestep
            t = torch.randint(
                0,
                diffusion.timesteps,
                (x0.size(0),),
                device=device
            )

            # Add noise
            noisy_x, noise = diffusion.add_noise(x0, t)

            # Predict noise
            predicted_noise = network(noisy_x, t, labels)

            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x0.size(0)

        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {epoch_loss:.4f}"
            )

    # Save trained model and required metadata
    checkpoint_path = os.path.join(
        output_dir,
        "diffusion_checkpoint.pt"
    )

    torch.save({
        "model_state_dict": network.state_dict(),
        "contour_mean": float(contour_mean),
        "contour_std": float(contour_std),
        "timesteps": timesteps,
        "channels": channels,
        "time_dim": time_dim
    }, checkpoint_path)

    # Save training loss
    np.save(
        os.path.join(output_dir, "training_loss.npy"),
        np.array(loss_history)
    )

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Diffusion Training Loss")
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()

    print("Training completed!")
    print("Checkpoint saved:", checkpoint_path)

    return checkpoint_path, loss_history