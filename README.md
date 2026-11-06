# Carnatic Musical Phrase Generation

A conditional 1D diffusion model for generating Carnatic musical pitch contours for selected phrase–raga combinations.

## Project structure

```text
Carnatic-Musical-Phrase-Generation/
├── preprocessing.py   # Dataset discovery and pitch-contour preparation
├── model.py           # Conditional 1D U-Net and Gaussian diffusion
├── train.py           # Training, generation and saving outputs
├── models/
└── notebooks/
```

## Run in Google Colab

Dataset location in Google Drive: `MyDrive/carnatic/Carnatic`.

### 1. Mount Drive and clone the repository

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/gauthamburg/Carnatic-Musical-Phrase-Generation.git
```

If already cloned in the current runtime, use `!git pull origin main` from inside the repository instead.

### 2. Set paths and import modules

```python
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = "/content/Carnatic-Musical-Phrase-Generation"
DATA_DIR = "/content/drive/MyDrive/carnatic/Carnatic"
OUTPUT_DIR = "/content/drive/MyDrive/carnatic_diffusion"

sys.path.insert(0, PROJECT_DIR)

from preprocessing import load_training_data
from model import ConditionalUNet1D, GaussianDiffusion1D
from train import train_and_generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

### 3. Prepare the dataset

```python
dataset, dataloader, pair_to_id, contour_mean, contour_std = \
    load_training_data(DATA_DIR, batch_size=16)
```

`load_training_data()` runs the preprocessing pipeline:
- `discover_pairs()` selects the four most frequent eligible phrase–raga pairs.
- `extract_contours()` extracts pitch segments, converts pitch to cents relative to tonic, interpolates missing values and resamples each contour to 128 points.
- `prepare_dataset()` encodes phrase–raga labels, normalizes contours and creates the PyTorch Dataset and DataLoader.

It returns the dataset, DataLoader, class-ID mapping, normalization mean and standard deviation. A typical batch has shape `[16, 1, 128]`.

### 4. Train and generate

```python
network, diffusion, loss_history, generated = train_and_generate(
    dataloader=dataloader,
    dataset=dataset,
    pair_to_id=pair_to_id,
    contour_mean=contour_mean,
    contour_std=contour_std,
    output_dir=OUTPUT_DIR,
    device=device,
    epochs=100,
    learning_rate=2e-4,
    batch_size=16,
    timesteps=500,
    seq_len=128,
    n_samples=4
)
```

This initializes the Conditional U-Net and diffusion process, trains the model, saves the checkpoint and loss plot, and generates four contours for each phrase–raga class. It returns the trained network, diffusion object, loss history and generated contours.

## Main configuration

| Parameter | Value |
|---|---:|
| Phrase–raga classes | 4 |
| Contour length | 128 points |
| U-Net base channels | 64 |
| Time embedding dimension | 128 |
| Diffusion timesteps | 500 |
| Training epochs | 100 |
| Batch size | 16 |
| Learning rate | 0.0002 |
| Generated contours per class | 4 |

## Saved outputs

Results are saved in `MyDrive/carnatic_diffusion/`:

| File | Description |
|---|---|
| `conditional_diffusion.pt` | Model weights, normalization statistics, class mapping and timestep count |
| `training_loss.npy` | Loss values across epochs |
| `training_loss.png` | Training loss plot |
| `generated_contours.npy` | Generated contours grouped by phrase–raga pair |

With four classes and four samples per class, the generated file contains 16 contours. Each contour has 128 pitch values in cents relative to the tonic. These are pitch contours, not audio files.
