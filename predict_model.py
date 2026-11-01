from pathlib import path

import numpy as np
import torch
from torch import nn

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


