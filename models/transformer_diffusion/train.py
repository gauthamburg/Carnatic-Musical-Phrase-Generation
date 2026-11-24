import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import TransformerDenoiser1D, GaussianDiffusion1D


def _get_contours_from_batch(batch):
    # Supports the project's current DataLoader returning x,
    # and also tuple/list batches such as (x, labels).
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    return x


def train_model(
    dataloader,
    contour_mean,
    contour_std,
    device,
    output_dir,
    epochs=100,
    learning_rate=2e-4,
    seq_len=128,
    timesteps=500,
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.1,
    time_dim=128
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    model_params = {
        "seq_len": seq_len,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "time_dim": time_dim
    }

    model = TransformerDenoiser1D(**model_params).to(device)
    diffusion = GaussianDiffusion1D(
        timesteps=timesteps, device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    loss_history = []

    print("Transformer-DDPM initialized")
    print("Device:", device)
    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters()))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch in dataloader:
            x0 = _get_contours_from_batch(batch).to(device).float()

            if x0.ndim == 2:
                x0 = x0.unsqueeze(1)
            elif x0.ndim == 3 and x0.shape[1] != 1 and x0.shape[-1] == 1:
                x0 = x0.transpose(1, 2)

            if x0.ndim != 3 or x0.shape[1] != 1:
                raise ValueError(
                    f"Expected contours [B,1,L], received {tuple(x0.shape)}"
                )

            batch_size = x0.size(0)
            t = torch.randint(
                0, diffusion.timesteps, (batch_size,), device=device
            )
            noisy_x, noise = diffusion.add_noise(x0, t)
            predicted_noise = model(noisy_x, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * batch_size
            sample_count += batch_size

        epoch_loss = running_loss / max(sample_count, 1)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    checkpoint_path = os.path.join(
        output_dir, "transformer_diffusion_checkpoint.pt"
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_params": model_params,
        "timesteps": timesteps,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "contour_mean": float(contour_mean),
        "contour_std": float(contour_std),
        "loss_history": loss_history,
        "model_name": "transformer_diffusion"
    }
    torch.save(checkpoint, checkpoint_path)

    loss_path = os.path.join(output_dir, "training_loss.png")
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Noise-prediction MSE")
    plt.title("Transformer-DDPM Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.show()
    plt.close()

    print("Checkpoint saved:", checkpoint_path)
    return checkpoint_path, loss_history
