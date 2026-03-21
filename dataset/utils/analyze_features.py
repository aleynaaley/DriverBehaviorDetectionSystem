import pandas as pd
from pathlib import Path

# =========================================================
# AYARLAR
# =========================================================
DATASET_ROOT = Path("../data")
INPUT_CSV = DATASET_ROOT / "../data/features_temporal.csv"

FEATURES_TO_CHECK = [
    "ear_ratio",
    "ear_diff",
    "mar_ratio",
    "mar_diff",
    "perclos_5s",
    "perclos_10s",
    "blink_rate_5s",
    "ear_std_5s",
    "ear_velocity",
    "abs_delta_pitch",
    "abs_delta_yaw",
    "abs_delta_pitch_mean_5s",
    "abs_delta_yaw_mean_5s",
]

# =========================================================
# VERİYİ OKU
# =========================================================
df = pd.read_csv(INPUT_CSV)

print("Shape:", df.shape)
print("\nLabel dağılımı:")
print(df["label"].value_counts())

# =========================================================
# SAFE vs DROWSY ÖZET
# =========================================================
summary_rows = []

for feat in FEATURES_TO_CHECK:
    if feat not in df.columns:
        print(f"[WARN] Kolon yok: {feat}")
        continue

    for label_val, label_name in [(0, "safe"), (1, "drowsy")]:
        part = df[df["label"] == label_val][feat]

        summary_rows.append({
            "feature": feat,
            "class": label_name,
            "count": int(part.count()),
            "mean": float(part.mean()),
            "median": float(part.median()),
            "std": float(part.std()),
            "min": float(part.min()),
            "max": float(part.max()),
        })

summary_df = pd.DataFrame(summary_rows)

print("\n================ FEATURE ÖZETİ ================\n")
for feat in summary_df["feature"].unique():
    sub = summary_df[summary_df["feature"] == feat]
    print(f"\n### {feat}")
    print(sub.to_string(index=False))

# =========================================================
# SINIFLAR ARASI FARK TABLOSU
# =========================================================
diff_rows = []

for feat in FEATURES_TO_CHECK:
    if feat not in df.columns:
        continue

    safe_part = df[df["label"] == 0][feat]
    drowsy_part = df[df["label"] == 1][feat]

    diff_rows.append({
        "feature": feat,
        "safe_mean": float(safe_part.mean()),
        "drowsy_mean": float(drowsy_part.mean()),
        "mean_diff(drowsy-safe)": float(drowsy_part.mean() - safe_part.mean()),
        "safe_median": float(safe_part.median()),
        "drowsy_median": float(drowsy_part.median()),
        "median_diff(drowsy-safe)": float(drowsy_part.median() - safe_part.median()),
    })

diff_df = pd.DataFrame(diff_rows)

print("\n================ SINIF FARKLARI ================\n")
print(diff_df.to_string(index=False))

# =========================================================
# CSV KAYDET
# =========================================================
summary_df.to_csv(DATASET_ROOT / "feature_summary_by_class.csv", index=False)
diff_df.to_csv(DATASET_ROOT / "feature_differences_safe_vs_drowsy.csv", index=False)

print("\nKaydedildi:")
print("-", DATASET_ROOT / "feature_summary_by_class.csv")
print("-", DATASET_ROOT / "feature_differences_safe_vs_drowsy.csv")