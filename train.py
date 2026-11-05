

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import ConditionalUNet1D, GaussianDiffusion1D



@torch.no_grad()
def generate_contours(network, diffusion, class_id,
                      n_samples=4, seq_len=128, device="cpu"):

    network.eval()

    # Start with random Gaussian noise
    x = torch.randn(n_samples, 1, seq_len, device=device)

    labels = torch.full(
        (n_samples,),
        class_id,
        dtype=torch.long,
        device=device
    )

    # Reverse diffusion: 499 -> 0
    for step in reversed(range(diffusion.timesteps)):

        t = torch.full(
            (n_samples,),
            step,
            device=device,
            dtype=torch.long
        )

        beta = diffusion.betas[step]
        alpha = diffusion.alphas[step]
        alpha_bar = diffusion.alpha_bars[step]

        # Predict noise using Conditional U-Net
        predicted_noise = network(x, t, labels)

        # Reverse diffusion equation
        x = (
            x - beta / torch.sqrt(1 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha)

        # Add noise except at the final step
        if step > 0:
            alpha_bar_prev = diffusion.alpha_bars[step - 1]

            variance = (
                beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
            )

            x += torch.sqrt(variance) * torch.randn_like(x)

    return x.squeeze(1).cpu().numpy()


# ==========================================
# 2. Train the Conditional Diffusion Model
# ==========================================

def train_model(network, diffusion, dataloader, dataset,
                device, epochs=100, learning_rate=2e-4):

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=learning_rate
    )

    loss_history = []

    network.train()

    for epoch in range(epochs):

        epoch_loss = 0.0

        for x0, labels in dataloader:

            x0 = x0.to(device)
            labels = labels.to(device)

            # Randomly select diffusion timestep
            t = torch.randint(
                0,
                diffusion.timesteps,
                (x0.size(0),),
                device=device
            )

            # Add noise to original contours
            noisy_x, noise = diffusion.add_noise(x0, t)

            # Predict the added noise
            predicted_noise = network(noisy_x, t, labels)

            # Calculate noise prediction loss
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x0.size(0)

        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {epoch_loss:.4f}"
            )

    return loss_history


# ==========================================
# 3. Complete Training and Generation Pipeline
# ==========================================

def train_and_generate(
    dataloader,
    dataset,
    pair_to_id,
    contour_mean,
    contour_std,
    output_dir,
    device,
    epochs=100,
    learning_rate=2e-4,
    batch_size=16,
    timesteps=500,
    seq_len=128,
    n_samples=4
):

    os.makedirs(output_dir, exist_ok=True)

    # Initialize Conditional U-Net
    network = ConditionalUNet1D(
        num_classes=len(pair_to_id),
        channels=64,
        time_dim=128
    ).to(device)

    # Initialize Gaussian Diffusion
    diffusion = GaussianDiffusion1D(
        timesteps=timesteps,
        device=device
    )

    print("Model initialized.")
    print("Device:", device)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in network.parameters())
    )

    # --------------------------------------
    # Train model
    # --------------------------------------

    print("\nStarting training...\n")

    loss_history = train_model(
        network,
        diffusion,
        dataloader,
        dataset,
        device,
        epochs,
        learning_rate
    )

    # --------------------------------------
    # Save trained checkpoint
    # --------------------------------------

    checkpoint_path = os.path.join(
        output_dir,
        "conditional_diffusion.pt"
    )

    torch.save({
        "model_state_dict": network.state_dict(),
        "contour_mean": contour_mean,
        "contour_std": contour_std,
        "pair_to_id": pair_to_id,
        "timesteps": diffusion.timesteps,
    }, checkpoint_path)

    print("\nModel saved:", checkpoint_path)

    # --------------------------------------
    # Save training loss
    # --------------------------------------

    loss_path = os.path.join(
        output_dir,
        "training_loss.npy"
    )

    np.save(loss_path, np.array(loss_history))

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE noise-prediction loss")
    plt.title("Conditional Diffusion Training Loss")
    plt.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(output_dir, "training_loss.png"),
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # --------------------------------------
    # Generate contours for each class
    # --------------------------------------

    print("\nStarting contour generation...\n")

    generated = {}

    for (phrase, raga), class_id in pair_to_id.items():

        samples = generate_contours(
            network,
            diffusion,
            class_id,
            n_samples=n_samples,
            seq_len=seq_len,
            device=device
        )

        # Convert normalized values back to cents
        samples = samples * contour_std + contour_mean

        generated[(phrase, raga)] = samples

        print(
            f"Generated {n_samples} contours for "
            f"{phrase} — {raga}"
        )

    # --------------------------------------
    # Save generated contours
    # --------------------------------------

    generated_path = os.path.join(
        output_dir,
        "generated_contours.npy"
    )

    np.save(
        generated_path,
        generated,
        allow_pickle=True
    )

    print("\nGenerated contours saved:", generated_path)

    return network, diffusion, loss_history, generated
