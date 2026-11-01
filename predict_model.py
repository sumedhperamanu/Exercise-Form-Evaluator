from pathlib import path

import numpy as np
import torch
from torch import nn

num_frames = 30
input_size = 132

device = torch.device(
    "cuda" if torch.cuda.is_available else "cpu"
)

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
