import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/aleyaley/Desktop/AB/model/sequence/class_weight/data")

X_TRAIN_PATH = DATA_DIR / "X_train_sequences.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train_sequences.npy"
X_TEST_PATH = DATA_DIR / "X_test_sequences.npy"
Y_TEST_PATH = DATA_DIR / "y_test_sequences.npy"

OUTPUT_MODEL = BASE_DIR / "model.pth"
OUTPUT_METRICS = BASE_DIR / "metrics.json"
OUTPUT_REPORT = BASE_DIR / "classification_report.txt"
OUTPUT_CM = BASE_DIR / "confusion_matrix.png"

# =========================================================
# PARAM
# =========================================================
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
LSTM_HIDDEN = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =========================================================
# DATA
# =========================================================
X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# normalization
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("\nNormalization applied ✔️")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# MODEL
# input: (B, T, F)
# conv1d wants: (B, F, T)
# =========================================================
class CNNLSTMModel(nn.Module):
    def __init__(self, input_features, lstm_hidden=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)          # (B, F, T)
        x = self.conv(x)                # (B, 128, T')
        x = x.permute(0, 2, 1)          # (B, T', 128)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = CNNLSTMModel(
    input_features=X_train.shape[2],
    lstm_hidden=LSTM_HIDDEN
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================================================
# TRAIN
# =========================================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# =========================================================
# TEST
# =========================================================
model.eval()
all_probs = []
all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())

y_true = np.array(all_true)
y_prob = np.array(all_probs)
y_pred = np.array(all_preds)

# =========================================================
# METRİKLER
# =========================================================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4)

metrics = {
    "model": "CNN_LSTM",
    "random_state": RANDOM_STATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "lstm_hidden": LSTM_HIDDEN,
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "roc_auc": roc,
    "confusion_matrix": cm.tolist(),
}

torch.save(model.state_dict(), OUTPUT_MODEL)

with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["safe", "drowsy"])
disp.plot(cmap="Blues")
plt.title("CNN+LSTM Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=200)
plt.close()

print("\n=== CNN+LSTM METRİKLER ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1       :", f1)
print("ROC AUC  :", roc)

print("\nClassification Report:")
print(report)

print("\nKaydedildi:")
print("-", OUTPUT_MODEL)
print("-", OUTPUT_METRICS)
print("-", OUTPUT_REPORT)
print("-", OUTPUT_CM)