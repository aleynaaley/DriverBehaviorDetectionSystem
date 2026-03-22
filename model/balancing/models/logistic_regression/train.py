import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================================================
# AYARLAR
# =========================================================
BASE_DIR = Path(__file__).resolve().parent                 # .../balancing/models/logistic_regression
BALANCING_DIR = BASE_DIR.parent.parent                     # .../balancing
BALANCED_DATA_DIR = BALANCING_DIR / "data/balanced_data"

TRAIN_CSV = BALANCED_DATA_DIR / "train_subject_split_balanced.csv"
TEST_CSV = BALANCED_DATA_DIR / "test_subject_split.csv"

OUTPUT_MODEL = BASE_DIR / "model.pkl"
OUTPUT_METRICS = BASE_DIR / "metrics.json"
OUTPUT_REPORT = BASE_DIR / "classification_report.txt"
OUTPUT_CM = BASE_DIR / "confusion_matrix.png"
OUTPUT_COEFS = BASE_DIR / "feature_coefficients.csv"

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

print("Balanced train shape:", train_df.shape)
print("Test shape          :", test_df.shape)

feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

X_train = train_df[feature_cols].copy()
y_train = train_df["label"].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df["label"].copy()

print("\nFeature sayısı:", len(feature_cols))
print("İlk birkaç feature:", feature_cols[:10])

# =========================================================
# MODEL
# =========================================================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=2000,
        solver="liblinear"
    ))
])

print("\nModel eğitiliyor...")
model.fit(X_train, y_train)

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
    "model": "LogisticRegression",
    "train_strategy": "balanced_undersampling",
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
plt.title("Logistic Regression - Confusion Matrix (Balanced Train)")
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=200)
plt.close()

clf = model.named_steps["clf"]
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": clf.coef_[0],
    "abs_coefficient": abs(clf.coef_[0]),
}).sort_values("abs_coefficient", ascending=False)

coef_df.to_csv(OUTPUT_COEFS, index=False)

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
print("-", OUTPUT_COEFS)