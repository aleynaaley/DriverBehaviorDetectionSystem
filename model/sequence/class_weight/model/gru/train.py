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
HIDDEN_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# SEED
# =========================================================
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

# =========================================================
# 🔥 NORMALIZATION (EN KRİTİK KISIM)
# SADECE TRAIN ÜZERİNDEN
# =========================================================
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("\nNormalization applied ✔️")

# =========================================================
# TENSOR
# =========================================================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =========================================================
# DATALOADER
# =========================================================
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# MODEL
# =========================================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = GRUModel(input_size=X_train.shape[2], hidden_size=HIDDEN_SIZE).to(DEVICE)

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

# =========================================================
# SAVE
# =========================================================
torch.save(model.state_dict(), OUTPUT_MODEL)

with open(OUTPUT_METRICS, "w") as f:
    json.dump({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    }, f, indent=2)

with open(OUTPUT_REPORT, "w") as f:
    f.write(report)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig(OUTPUT_CM)
plt.close()

# =========================================================
# PRINT
# =========================================================
print("\n=== GRU (NORMALIZED) ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1       :", f1)
print("ROC AUC  :", roc)