
# ==========================================
# PREPROCESSING MODULE
# ==========================================

import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 128


# ==========================================
# 1. Read raga metadata
# ==========================================

def get_performance_ragas(folder):

    ragas = set()

    for filename in os.listdir(folder):

        if not filename.endswith(".json"):
            continue

        try:
            with open(
                os.path.join(folder, filename),
                "r",
                encoding="utf-8"
            ) as f:
                metadata = json.load(f)

            raga_data = metadata.get("raaga", [])

            if isinstance(raga_data, list):

                for item in raga_data:

                    if isinstance(item, dict) and item.get("name"):
                        ragas.add(item["name"].strip())

            elif isinstance(raga_data, str) and raga_data.strip():

                ragas.add(raga_data.strip())

        except (json.JSONDecodeError, OSError):
            continue

    return ragas


# ==========================================
# 2. Pitch extraction
# ==========================================

def get_tonic(folder):

    for filename in os.listdir(folder):

        if filename.endswith(".ctonic.txt"):

            try:
                with open(
                    os.path.join(folder, filename),
                    "r"
                ) as f:

                    return float(f.read().strip().split()[0])

            except (ValueError, OSError, IndexError):
                continue

    return None


def load_pitch(folder):

    # Prefer timestamped pitch data
    for filename in os.listdir(folder):

        if filename.endswith(".pitch.txt"):

            try:
                data = np.loadtxt(
                    os.path.join(folder, filename)
                )

                if data.ndim == 2 and data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]

            except (ValueError, OSError):
                continue

    # Fall back to pitch array
    for filename in os.listdir(folder):

        if filename.endswith("_pitch.npy"):

            try:
                pitch = np.load(
                    os.path.join(folder, filename)
                ).astype(float)

                duration = None

                for jf in os.listdir(folder):

                    if jf.endswith(".json"):

                        try:
                            with open(
                                os.path.join(folder, jf),
                                "r"
                            ) as f:

                                metadata = json.load(f)

                            duration = (
                                float(metadata.get("length", 0)) / 1000
                            )

                            if duration > 0:
                                break

                        except (ValueError, OSError):
                            continue

                if duration and len(pitch) > 1:

                    times = np.linspace(
                        0, duration, len(pitch)
                    )

                    return times, pitch

            except (ValueError, OSError):
                continue

    return None, None


# ==========================================
# 3. Extract selected phrase–raga contours
# ==========================================

def extract_contours(
    DATA_DIR,
    raga,
    phrase,
    seq_len=SEQ_LEN
):

    contours = []
    skipped = 0
    matching_performances = 0

    for root, dirs, files in os.walk(DATA_DIR):

        annotation_files = [
            f for f in files
            if f.endswith(".mphrases-manual.txt")
        ]

        if not annotation_files:
            continue

        # Check whether this performance belongs to the selected raga
        performance_ragas = get_performance_ragas(root)

        matching_raga = any(
            name.strip().casefold() == raga.strip().casefold()
            for name in performance_ragas
        )

        if not matching_raga:
            continue

        matching_performances += 1

        # Load pitch and tonic
        times, f0 = load_pitch(root)

        if times is None:
            continue

        tonic = get_tonic(root)

        if tonic is None or tonic <= 0:
            continue

        # Identify valid pitch values
        valid = np.isfinite(f0) & (f0 > 0)

        if valid.sum() < 2:
            continue

        # Convert Hz to cents relative to tonic
        cents = np.full(len(f0), np.nan)

        cents[valid] = (
            1200 * np.log2(f0[valid] / tonic)
        )

        # Interpolate missing pitch values
        cents = pd.Series(cents).interpolate(
            limit_direction="both"
        ).to_numpy()

        # Process annotation files
        for af in annotation_files:

            try:
                df = pd.read_csv(
                    os.path.join(root, af),
                    sep=r"\s+",
                    header=None,
                    names=[
                        "start",
                        "unused",
                        "duration",
                        "phrase"
                    ],
                    engine="python"
                )

                # Select only the requested phrase
                matches = df[
                    df["phrase"].astype(str).str.strip()
                    == phrase.strip()
                ]

                for _, row in matches.iterrows():

                    try:
                        start = float(row["start"])
                        duration = float(row["duration"])

                        if duration <= 0:
                            skipped += 1
                            continue

                        # Extract pitch segment
                        mask = (
                            (times >= start) &
                            (times <= start + duration)
                        )

                        segment_t = times[mask]
                        segment_y = cents[mask]

                        good = np.isfinite(segment_y)

                        if good.sum() < 2:
                            skipped += 1
                            continue

                        relative_t = segment_t[good] - start
                        relative_y = segment_y[good]

                        # Resample to fixed length
                        target_t = np.linspace(
                            0, duration, seq_len
                        )

                        contour = np.interp(
                            target_t,
                            relative_t,
                            relative_y
                        )

                        contours.append(
                            contour.astype(np.float32)
                        )

                    except (ValueError, TypeError):
                        skipped += 1
                        continue

            except Exception as e:
                print("Skipped annotation:", af, "|", e)

    if len(contours) == 0:

        raise ValueError(
            f"No pitch contours found for "
            f"Raga='{raga}', Phrase='{phrase}'. "
            f"Check the names and available pitch files."
        )

    contours = np.array(contours, dtype=np.float32)

    print("\nExtraction complete")
    print("Selected raga:", raga)
    print("Selected phrase:", phrase)
    print("Matching performances:", matching_performances)
    print("Extracted phrase instances:", len(contours))
    print("Skipped instances:", skipped)
    print("Contour array shape:", contours.shape)

    return contours


# ==========================================
# 4. PyTorch Dataset
# ==========================================

class PhraseDataset(Dataset):

    def __init__(self, data):

        self.data = torch.tensor(
            data,
            dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


# ==========================================
# 5. Prepare selected dataset
# ==========================================

def prepare_dataset(
    contours,
    batch_size=16
):

    # Calculate normalization statistics
    contour_mean = float(contours.mean())
    contour_std = float(contours.std())

    normalized_contours = (
        contours - contour_mean
    ) / (contour_std + 1e-6)

    dataset = PhraseDataset(normalized_contours)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print("\nDataset prepared")
    print("Dataset size:", len(dataset))
    print("Contour shape:", dataset[0].shape)
    print("Batch size:", batch_size)
    print("Normalization mean:", round(contour_mean, 2))
    print("Normalization std:", round(contour_std, 2))

    return dataset, dataloader, contour_mean, contour_std


# ==========================================
# 6. Main data-loading function
# ==========================================

def load_training_data(
    DATA_DIR,
    raga,
    phrase,
    batch_size=16,
    seq_len=SEQ_LEN
):

    contours = extract_contours(
        DATA_DIR=DATA_DIR,
        raga=raga,
        phrase=phrase,
        seq_len=seq_len
    )

    return prepare_dataset(
        contours=contours,
        batch_size=batch_size
    )
