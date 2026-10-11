from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

data_dir = Path("data/squat")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

num_frames = 30
num_landmarks = 33
values_per_landmark = 4
input_size = num_landmarks * values_per_landmark

batch_size = 16
epochs = 60
learning_rate = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)

def load_rep(csv_path):
    df = pd.read_csv(csv_path, skiprows=2)

    expected_cols = [
        f"{axis}_{i}"
        for i in range(num_landmarks)
        for axis in ("x", "y", "z", "visibility")
    ]

    missing = set(expected_cols) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing landmark columns in {csv_path}"
            f"{sorted(missing)}"
        )

    sequence = df[expected_cols].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    sequence[~np.isfinite(sequence)] = np.nan

    sequence = (
        pd.DataFrame(sequence)
        .interpolate(axis=0, limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    if len(sequence) < 2:
        raise ValueError("Rep has too few frames")

    old_positions = np.linspace(0.0, 1.0, len(sequence))
    new_positions = np.linspace(0.0, 1.0, num_frames)

    resized = np.stack([
        np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_idx]
        )
        for feature_idx in range(input_size)
    ], axis=1)

    return resized

sequences = []
labels = []
files = []

for label_dir in sorted(data_dir.iterdir()):
    if not label_dir.is_dir:
        continue