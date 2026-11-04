from pathlib import Path

import numpy as np
import torch
from torch import nn
import pandas as pd

num_frames = 30
input_size = 132

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Defining MLP architecture to classify squat sequences

class SquatClassifier (nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * num_frames, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Preparing model for inference

checkpoint = torch.load(
    "models/squat_classifier.pt",
    map_location = device,
    weights_only = True
)

classes = checkpoint["classes"]

model = SquatClassifier(
    input_size, len(classes)
).to(device)

model.load_state_dict(checkpoint["state_dict"])
model.eval()

feature_mean = checkpoint["feature_mean"].cpu().numpy()
feature_std = checkpoint["feature_std"].cpu().numpy()

def predict_rep(frames):
    """Return predicted labels and associated probabilities."""

    sequence = np.asarray(frames, dtype=np.float32).copy()

    if sequence.ndim != 2 or sequence.shape[1] != input_size:
        raise ValueError("Expected shape (frames, 132)")

    if len(sequence) < 2:
        return {
            "label": "unable_to_assess",
            "probabilities": {}
        }

    sequence[~np.isfinite(sequence)] = np.nan

    # Preprocessing + classification of recorded squat reps

    sequence = (
        pd.DataFrame(sequence)
        .interpolate(axis=0, limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    old_positions = np.linspace(0, 1, len(sequence))
    new_positions = np.linspace(0, 1, num_frames)

    sequence = np.stack([
        np.interp(
            new_positions,
            old_positions,
            sequence[:, j]
        )
        for j in range(input_size)
    ], axis=1)

    sequence = (sequence - feature_mean) / feature_std

    x = torch.tensor(
        sequence, dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(x), dim=1
        )[0].cpu().numpy()

    best_idx = int(np.argmax(probabilities))

    return {
        "label": classes[best_idx],
        "probabilities": {
            name: float(prob)
            for name, prob in zip(classes, probabilities)
        }
    }

def predict_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=2)

    expected_cols = [
            f"{axis}_{i}"
            for i in range(33)
            for axis in ("x", "y", "z", "visibility")
    ]

    missing_cols = [
        col for col in expected_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"CSV is missing columns: {missing_cols} in {csv_path}"
        )

    frames = df[expected_cols].to_numpy(
        dtype=np.float32,
        copy=True
    )

    return predict_rep(frames)

testing_dir = Path("data/squat/testing_data")

for category in ["good", "shallow", "knee_valgus"]:
    folder = testing_dir / category

    for csv_path in sorted(folder.glob("*.csv")):
        result = predict_csv(csv_path)

        print(f"\n File: {csv_path.name}")
        print(f" Actual Class: {category}")
        print(f"Predicted Class: {result['label']}")

        for label, probability in result["probabilities"].items():
            print(f"{label}: {probability:.2%}")

