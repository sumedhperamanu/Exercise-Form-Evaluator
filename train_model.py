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
    if not label_dir.is_dir():
        continue

    label = label_dir.name

    for csv_path in sorted(label_dir.glob("*.csv")):
        try:
            sequence = load_rep(csv_path)

            sequences.append(sequence)
            labels.append(label)
            files.append(str(csv_path))

        except Exception as exc:
            print(f"Skipping {csv_path}: {exc}")

if not sequences:
    raise RuntimeError(
        f"No usable reps found in {data_dir.resolve()}"
    )

X = np.stack(sequences).astype(np.float32)
labels = np.asarray(labels)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

print("Dataset shape", X.shape)
print("Classes", encoder.classes_)
print("Reps per class: ")
for class_name in encoder.classes_:
    print(f"    {class_name}: {(labels == class_name).sum()}")

label_counts = np.bincount(y)

if len(label_counts) < 2 or np.min(label_counts) < 2:
    raise ValueError(
        "Each label needs at least two reps for a stratified train/test split"
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

feature_mean = X_train.reshape(-1, input_size).mean(axis=0)
feature_std = X_train.reshape(-1, input_size).std(axis=0)

feature_std[feature_std < 1e-6] = 1

X_train = (X_train - feature_mean)/(feature_std)
X_test = (X_test - feature_mean)/(feature_std)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size = batch_size,
    shuffle = True
)

class SquatClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten,

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

model = SquatClassifier(
    input_size,
    len(encoder.classes_)
).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = learning_rate,
    weight_decay = 1e^-4
)