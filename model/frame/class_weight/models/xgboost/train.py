import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
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
# AYARLAR
# =========================================================
BASE_DIR = Path(__file__).resolve().parent                 # .../class_weight/models/xgboost
CLASS_WEIGHT_DIR = BASE_DIR.parent.parent                  # .../class_weight
SPLITS_DIR = CLASS_WEIGHT_DIR / "splits"

TRAIN_CSV = SPLITS_DIR / "train_subject_split.csv"
TEST_CSV = SPLITS_DIR / "test_subject_split.csv"

OUTPUT_MODEL = BASE_DIR / "model.pkl"
OUTPUT_METRICS = BASE_DIR / "metrics.json"
OUTPUT_REPORT = BASE_DIR / "classification_report.txt"
OUTPUT_CM = BASE_DIR / "confusion_matrix.png"
OUTPUT_IMPORTANCE = BASE_DIR / "feature_importance.csv"

RANDOM_STATE = 42

DROP_COLS = [
    "person_id",
    "video_id",
    "segment_type",
    "frame_id",
    "timestamp_sec",
    "fps",
    "label",
]

# =========================================================
# KONTROLLER
# =========================================================
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Train dosyası bulunamadı: {TRAIN_CSV}")

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test dosyası bulunamadı: {TEST_CSV}")

# =========================================================
# VERİYİ OKU
# =========================================================
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

X_train = train_df[feature_cols].copy()
y_train = train_df["label"].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df["label"].copy()

print("\nFeature sayısı:", len(feature_cols))
print("İlk birkaç feature:", feature_cols[:10])

# =========================================================
# CLASS IMBALANCE AĞIRLIĞI
# XGBoost'ta class_weight yerine scale_pos_weight kullanıyoruz
# Pozitif sınıf = 1 (drowsy)
# Negatif sınıf = 0 (safe)
# =========================================================
n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()

# Burada amaç safe sınıfını güçlendirmek olduğundan sample_weight kullanacağız
# safe=0 azınlık olduğu için ona daha büyük ağırlık vereceğiz
sample_weight = y_train.map({
    0: len(y_train) / (2 * n_negative),
    1: len(y_train) / (2 * n_positive),
}).values

print("\nTrain label dağılımı:")
print(y_train.value_counts())
print("\nSample weight:")
print(f"class 0 weight = {len(y_train) / (2 * n_negative):.4f}")
print(f"class 1 weight = {len(y_train) / (2 * n_positive):.4f}")

# =========================================================
# MODEL
# =========================================================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nModel eğitiliyor...")
model.fit(X_train, y_train, sample_weight=sample_weight)

# =========================================================
# TAHMİN
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================================================
# METRİKLER
# =========================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

metrics = {
    "model": "XGBClassifier",
    "train_strategy": "class_weight_via_sample_weight",
    "random_state": RANDOM_STATE,
    "n_features": len(feature_cols),
    "train_shape": list(train_df.shape),
    "test_shape": list(test_df.shape),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
    "class_0_weight": float(len(y_train) / (2 * n_negative)),
    "class_1_weight": float(len(y_train) / (2 * n_positive)),
}

# =========================================================
# KAYDET
# =========================================================
joblib.dump(model, OUTPUT_MODEL)

with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["safe", "drowsy"])
disp.plot(cmap="Blues")
plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=200)
plt.close()

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(OUTPUT_IMPORTANCE, index=False)

# =========================================================
# EKRAN
# =========================================================
print("\n=== METRİKLER ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\n=== CONFUSION MATRIX ===")
print(cm)

print("\n=== CLASSIFICATION REPORT ===")
print(report)

print("\nKaydedilen dosyalar:")
print("-", OUTPUT_MODEL)
print("-", OUTPUT_METRICS)
print("-", OUTPUT_REPORT)
print("-", OUTPUT_CM)
print("-", OUTPUT_IMPORTANCE)