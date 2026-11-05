# Setup

import os
import json
import numpy as np
import pandas as pd

from collections import Counter

DATA_DIR = "/content/drive/MyDrive/carnatic/Carnatic"

SEQ_LEN = 128




# Cell 2: Discover frequent phrase–raga pairs

from collections import Counter

pair_counts = Counter()
pair_with_pitch = Counter()
pair_performances = {}

for root, dirs, files in os.walk(DATA_DIR):
    annotation_files = [
        f for f in files if f.endswith(".mphrases-manual.txt")
    ]
    json_files = [f for f in files if f.endswith(".json")]

    if not annotation_files or not json_files:
        continue

    # Find raga metadata
    ragas = set()

    for jf in json_files:
        try:
            with open(os.path.join(root, jf), "r", encoding="utf-8") as f:
                metadata = json.load(f)

            raga_data = metadata.get("raaga", [])

            if isinstance(raga_data, list):
                for item in raga_data:
                    if isinstance(item, dict) and item.get("name"):
                        ragas.add(item["name"].strip())
            elif isinstance(raga_data, str) and raga_data.strip():
                ragas.add(raga_data.strip())

        except (json.JSONDecodeError, OSError):
            pass

    # Skip performances without recorded raga metadata
    if not ragas:
        continue

    pitch_files = [
        f for f in files
        if f.endswith(".pitch.txt") or f.endswith("_pitch.npy")
    ]

    for af in annotation_files:
        try:
            df = pd.read_csv(
                os.path.join(root, af),
                sep=r"\s+",
                header=None,
                names=["start", "unused", "duration", "phrase"]
            )

            for phrase in df["phrase"].dropna().astype(str):
                for raga in ragas:
                    pair = (phrase, raga)
                    pair_counts[pair] += 1
                    pair_performances.setdefault(pair, set()).add(root)

                    if pitch_files:
                        pair_with_pitch[pair] += 1

        except Exception as e:
            print("Skipped:", af, "|", e)

print("Unique phrase–raga pairs:", len(pair_counts))

print("\nTOP 15 PAIRS BY FREQUENCY")
print("-" * 75)
print(f"{'Phrase':20} {'Raga':20} {'Total':>7} {'With pitch file':>16}")

for (phrase, raga), count in pair_counts.most_common(15):
    print(
        f"{phrase:20} {raga:20} {count:7} "
        f"{pair_with_pitch[(phrase, raga)]:16}"
    )


# Cell 3: Select top 4 phrase–raga pairs with pitch files

eligible_pairs = [
    (pair, count)
    for pair, count in pair_counts.most_common()
    if pair_with_pitch[pair] > 0
]

TOP_PAIRS = eligible_pairs[:4]

print("Selected phrase–raga pairs:\n")

for i, ((phrase, raga), count) in enumerate(TOP_PAIRS, start=1):
    print(
        f"{i}. Phrase: {phrase:15} | "
        f"Raga: {raga:15} | "
        f"Occurrences: {count} | "
        f"With pitch files: {pair_with_pitch[(phrase, raga)]}"
    )




SEQ_LEN = 128
contours = []
phrase_labels = []
raga_labels = []
skipped = 0

def get_tonic(folder):
    for f in os.listdir(folder):
        if f.endswith(".ctonic.txt"):
            try:
                with open(os.path.join(folder, f), "r") as file:
                    return float(file.read().strip().split()[0])
            except:
                pass
    return None

def load_pitch(folder):
    # Prefer timestamped pitch data
    for f in os.listdir(folder):
        if f.endswith(".pitch.txt"):
            data = np.loadtxt(os.path.join(folder, f))
            if data.ndim == 2 and data.shape[1] >= 2:
                return data[:, 0], data[:, 1]

    # Fall back to pitch array
    for f in os.listdir(folder):
        if f.endswith("_pitch.npy"):
            pitch = np.load(os.path.join(folder, f)).astype(float)

            # Estimate duration from JSON metadata
            duration = None
            for jf in os.listdir(folder):
                if jf.endswith(".json"):
                    try:
                        with open(os.path.join(folder, jf), "r") as file:
                            metadata = json.load(file)
                        duration = float(metadata.get("length", 0)) / 1000
                        if duration > 0:
                            break
                    except:
                        pass

            if duration and len(pitch) > 1:
                times = np.linspace(0, duration, len(pitch))
                return times, pitch

    return None, None

for root, dirs, files in os.walk(DATA_DIR):
    annotation_files = [
        f for f in files if f.endswith(".mphrases-manual.txt")
    ]
    if not annotation_files:
        continue

    # Identify the performance's raga
    ragas = set()
    for jf in files:
        if jf.endswith(".json"):
            try:
                with open(os.path.join(root, jf), "r", encoding="utf-8") as file:
                    metadata = json.load(file)
                rd = metadata.get("raaga", [])
                if isinstance(rd, list):
                    ragas.update(
                        item["name"].strip()
                        for item in rd
                        if isinstance(item, dict) and item.get("name")
                    )
                elif isinstance(rd, str) and rd.strip():
                    ragas.add(rd.strip())
            except:
                pass

    times, f0 = load_pitch(root)
    if times is None:
        continue

    tonic = get_tonic(root)
    if tonic is None or tonic <= 0:
        continue

    valid = np.isfinite(f0) & (f0 > 0)
    if valid.sum() < 2:
        continue

    # Convert pitch to cents relative to tonic; interpolate unvoiced regions
    cents = np.full(len(f0), np.nan)
    cents[valid] = 1200 * np.log2(f0[valid] / tonic)
    cents = pd.Series(cents).interpolate(limit_direction="both").to_numpy()

    for af in annotation_files:
        try:
            df = pd.read_csv(
                os.path.join(root, af),
                sep=r"\s+",
                header=None,
                names=["start", "unused", "duration", "phrase"],
                engine="python"
            )

            for (phrase, raga), _count in TOP_PAIRS:
                if raga not in ragas:
                    continue

                matches = df[
                    df["phrase"].astype(str).str.strip() == phrase
                ]

                for _, row in matches.iterrows():
                    start = float(row["start"])
                    duration = float(row["duration"])
                    if duration <= 0:
                        skipped += 1
                        continue

                    mask = (times >= start) & (times <= start + duration)
                    segment_t = times[mask]
                    segment_y = cents[mask]

                    good = np.isfinite(segment_y)
                    if good.sum() < 2:
                        skipped += 1
                        continue

                    # Interpolate and resample to 128 points
                    relative_t = segment_t[good] - start
                    relative_y = segment_y[good]
                    target_t = np.linspace(0, duration, SEQ_LEN)

                    contour = np.interp(
                        target_t, relative_t, relative_y
                    )
                    contours.append(contour.astype(np.float32))
                    phrase_labels.append(phrase)
                    raga_labels.append(raga)

        except Exception as e:
            print("Skipped annotation:", af, "|", e)

contours = np.array(contours, dtype=np.float32)

print("Extraction complete")
print("Contour array shape:", contours.shape)
print("Extracted samples:", len(contours))
print("Skipped phrase instances:", skipped)
print("Samples per phrase–raga pair:")
for (phrase, raga), _ in TOP_PAIRS:
    count = sum(
        p == phrase and r == raga
        for p, r in zip(phrase_labels, raga_labels)
    )
    print(f"{phrase:15} | {raga:15} | {count}")

