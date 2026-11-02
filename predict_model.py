from pathlib import path

import numpy as np
import torch
from torch import nn
import pandas as pd

num_frames = 30
input_size = 132

device = torch.device(
    "cuda" if torch.cuda.is_available else "cpu"
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
    "model/squats_classifier.pt",
    map_location = device,
    weights_only = True
)

classes = checkpoint["classes"]

model = SquatClassifier(
    input_size, len(classes)
).to(device)

model.load_state_dict(checkpoint["state_dict"])
model.eval()

feature_mean = checkpoint["feature_mean"]
feature_std = checkpoint["feature_std"]

def predict_rep(frames):
    """Return predicted labels and associated probabilities."""

    sequence = np.asarray(frames, dtype=np.float32)

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
        .fillna(0,0)
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

    best_idx = int(np.argmax[probabilities])

    return {
        "label": classes[best_idx],
        "probabilities": {
            name: float(prob)
            for name, prob in zip(classes, probabilities)
        }
    }


