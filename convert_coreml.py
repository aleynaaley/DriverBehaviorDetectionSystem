"""
BiLSTM → CoreML Dönüşümü
=========================
Çalıştırma:
  cd /Users/aleyaley/Desktop/AB
  source venv/bin/activate
  python convert_coreml.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from pathlib import Path

# ── Yollar ───────────────────────────────────────────────────
BASE      = Path("/Users/aleyaley/Desktop/AB")
DATA_DIR  = BASE / "model/sequence/class_weight/data"
MODEL_PTH = BASE / "model/sequence/class_weight/model/bilstm/model.pth"
CONFIG    = DATA_DIR / "sequence_config.json"
MEAN_NPY  = DATA_DIR / "train_mean.npy"
STD_NPY   = DATA_DIR / "train_std.npy"
OUT_DIR   = BASE / "ios_model"
OUT_DIR.mkdir(exist_ok=True)
OUT_MODEL = OUT_DIR / "DrowsinessModel.mlpackage"

# ── Model mimarisi (train.py ile aynı) ───────────────────────
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return torch.sigmoid(out)   # inference'da sigmoid ekle


def convert():
    # Config yükle
    cfg           = json.load(open(CONFIG))
    feature_names = cfg["feature_columns"]
    window_size   = cfg["window_size"]   # 30
    n_features    = cfg["n_features"]    # 51

    # Normalization parametreleri
    mean = np.load(MEAN_NPY).astype(np.float32)
    std  = np.load(STD_NPY).astype(np.float32)

    print(f"window_size : {window_size}")
    print(f"n_features  : {n_features}")
    print(f"mean[:3]    : {mean[:3]}")
    print(f"std[:3]     : {std[:3]}")

    # Model yükle
    model = BiLSTMModel(input_size=n_features, hidden_size=128)
    model.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
    model.eval()
    print("Model yüklendi ✓")

    # Trace et
    dummy = torch.randn(1, window_size, n_features)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    print("Trace edildi ✓")

    # CoreML'e çevir
    print("CoreML'e dönüştürülüyor...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="sequence",
            shape=(1, window_size, n_features),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="drowsiness_prob")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    # Metadata — Swift tarafında kullanılacak
    mlmodel.short_description = "Vigilance — Sürücü Yorgunluk Tespiti"
    mlmodel.author            = "Vigilance App"
    mlmodel.version           = "1.0"
    mlmodel.user_defined_metadata["feature_columns"] = json.dumps(feature_names)
    mlmodel.user_defined_metadata["window_size"]     = str(window_size)
    mlmodel.user_defined_metadata["n_features"]      = str(n_features)
    mlmodel.user_defined_metadata["train_mean"]      = json.dumps(mean.tolist())
    mlmodel.user_defined_metadata["train_std"]       = json.dumps(std.tolist())
    mlmodel.user_defined_metadata["threshold"]       = "0.5"

    mlmodel.save(str(OUT_MODEL))
    print(f"\n✓ Kaydedildi → {OUT_MODEL}")
    print(f"  window_size    : {window_size}")
    print(f"  n_features     : {n_features}")
    print(f"  feature sayısı : {len(feature_names)}")
    print(f"\nBu klasörü Xcode'a sürükle: {OUT_MODEL}")


if __name__ == "__main__":
    # coremltools kurulu değilse:
    # pip install coremltools
    convert()