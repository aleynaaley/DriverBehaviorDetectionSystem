import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# AYARLAR
# =========================================================
DATASET_ROOT = Path("../data")
INPUT_CSV = DATASET_ROOT / "features_raw.csv"
OUTPUT_CSV = DATASET_ROOT / "features_temporal.csv"

# PERCLOS ve blink için relatif EAR eşiği
EAR_RATIO_CLOSED_THR = 0.75

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def angle_diff_deg(a, b):
    """
    Açısal farkı -180..180 aralığına sarar.
    """
    diff = a - b
    return (diff + 180.0) % 360.0 - 180.0

def safe_divide(a, b):
    """
    Sıfıra bölünmeyi güvenli yapar.
    """
    b = np.where(np.abs(b) < 1e-12, np.nan, b)
    return a / b

def rolling_mean_min1(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rolling_std_min1(series, window):
    return series.rolling(window=window, min_periods=1).std().fillna(0.0)

def rolling_sum_min1(series, window):
    return series.rolling(window=window, min_periods=1).sum()

# =========================================================
# VERİYİ OKU
# =========================================================
df = pd.read_csv(INPUT_CSV)

# Çok önemli: kişi + video + segment + frame sıralaması
df = df.sort_values(
    ["person_id", "video_id", "segment_type", "frame_id"]
).reset_index(drop=True)

# =========================================================
# 1) BASIC TÜREV FEATURE'LAR
# =========================================================
df["ear_ratio"] = safe_divide(df["ear"], df["baseline_ear"])
df["ear_diff"] = df["ear"] - df["baseline_ear"]

df["mar_ratio"] = safe_divide(df["mar"], df["baseline_mar"])
df["mar_diff"] = df["mar"] - df["baseline_mar"]

df["delta_pitch"] = angle_diff_deg(df["pitch"].values, df["baseline_pitch"].values)
df["delta_yaw"] = angle_diff_deg(df["yaw"].values, df["baseline_yaw"].values)
df["delta_roll"] = angle_diff_deg(df["roll"].values, df["baseline_roll"].values)

df["abs_delta_pitch"] = np.abs(df["delta_pitch"])
df["abs_delta_yaw"] = np.abs(df["delta_yaw"])

# =========================================================
# 2) PERCLOS / BLINK İÇİN ARA FEATURE
# Hard EAR değil, relatif EAR
# =========================================================
df["eye_closed"] = (df["ear_ratio"] < EAR_RATIO_CLOSED_THR).astype(int)

# =========================================================
# 3) GRUP BAZLI TEMPORAL FEATURE'LAR
# Her safe/drowsy videosu bağımsız işlenir
# =========================================================
group_cols = ["person_id", "video_id", "segment_type"]
result_groups = []

for _, g in df.groupby(group_cols, sort=False):
    g = g.copy().reset_index(drop=True)

    fps = float(g["fps"].iloc[0]) if len(g) > 0 else 30.0
    if fps <= 0:
        fps = 30.0

    # Pencere uzunlukları (frame cinsinden)
    win_1s = max(1, int(round(1 * fps)))
    win_5s = max(1, int(round(5 * fps)))
    win_10s = max(1, int(round(10 * fps)))

    # -----------------------------------------------------
    # EAR rolling
    # -----------------------------------------------------
    g["ear_mean_1s"] = rolling_mean_min1(g["ear"], win_1s)
    g["ear_mean_5s"] = rolling_mean_min1(g["ear"], win_5s)
    g["ear_mean_10s"] = rolling_mean_min1(g["ear"], win_10s)

    # -----------------------------------------------------
    # MAR rolling
    # -----------------------------------------------------
    g["mar_mean_1s"] = rolling_mean_min1(g["mar"], win_1s)
    g["mar_mean_5s"] = rolling_mean_min1(g["mar"], win_5s)
    g["mar_mean_10s"] = rolling_mean_min1(g["mar"], win_10s)

    # -----------------------------------------------------
    # Head pose rolling
    # -----------------------------------------------------
    g["pitch_mean_1s"] = rolling_mean_min1(g["pitch"], win_1s)
    g["pitch_mean_5s"] = rolling_mean_min1(g["pitch"], win_5s)
    g["pitch_mean_10s"] = rolling_mean_min1(g["pitch"], win_10s)

    g["yaw_mean_1s"] = rolling_mean_min1(g["yaw"], win_1s)
    g["yaw_mean_5s"] = rolling_mean_min1(g["yaw"], win_5s)
    g["yaw_mean_10s"] = rolling_mean_min1(g["yaw"], win_10s)

    g["roll_mean_1s"] = rolling_mean_min1(g["roll"], win_1s)
    g["roll_mean_5s"] = rolling_mean_min1(g["roll"], win_5s)
    g["roll_mean_10s"] = rolling_mean_min1(g["roll"], win_10s)

    # -----------------------------------------------------
    # Mutlak head deviation rolling
    # -----------------------------------------------------
    g["abs_delta_pitch_mean_1s"] = rolling_mean_min1(g["abs_delta_pitch"], win_1s)
    g["abs_delta_pitch_mean_5s"] = rolling_mean_min1(g["abs_delta_pitch"], win_5s)
    g["abs_delta_pitch_mean_10s"] = rolling_mean_min1(g["abs_delta_pitch"], win_10s)

    g["abs_delta_yaw_mean_1s"] = rolling_mean_min1(g["abs_delta_yaw"], win_1s)
    g["abs_delta_yaw_mean_5s"] = rolling_mean_min1(g["abs_delta_yaw"], win_5s)
    g["abs_delta_yaw_mean_10s"] = rolling_mean_min1(g["abs_delta_yaw"], win_10s)

    # -----------------------------------------------------
    # PERCLOS
    # eye_closed 0/1 -> rolling mean = oran
    # -----------------------------------------------------
    g["perclos_5s"] = rolling_mean_min1(g["eye_closed"], win_5s)
    g["perclos_10s"] = rolling_mean_min1(g["eye_closed"], win_10s)

    # -----------------------------------------------------
    # Blink başlangıcı: 0 -> 1 geçişi
    # -----------------------------------------------------
    prev_closed = g["eye_closed"].shift(1).fillna(0).astype(int)
    g["blink_start"] = ((g["eye_closed"] == 1) & (prev_closed == 0)).astype(int)

    g["blink_count_5s"] = rolling_sum_min1(g["blink_start"], win_5s)
    g["blink_rate_5s"] = g["blink_count_5s"] / 5.0

    # -----------------------------------------------------
    # Std feature'lar
    # -----------------------------------------------------
    g["ear_std_5s"] = rolling_std_min1(g["ear"], win_5s)
    g["mar_std_5s"] = rolling_std_min1(g["mar"], win_5s)

    # -----------------------------------------------------
    # Velocity feature'lar
    # -----------------------------------------------------
    g["ear_velocity"] = g["ear"].diff().fillna(0.0)
    g["pitch_velocity"] = g["delta_pitch"].diff().fillna(0.0)
    g["yaw_velocity"] = g["delta_yaw"].diff().fillna(0.0)

    result_groups.append(g)

# =========================================================
# GRUPLARI BİRLEŞTİR
# =========================================================
out_df = pd.concat(result_groups, ignore_index=True)

# Debug için ara sütunları şimdilik tutuyoruz.
# out_df = out_df.drop(columns=["eye_closed", "blink_start"])

# =========================================================
# KAYDET
# =========================================================
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Bitti: {OUTPUT_CSV}")
print("Shape:", out_df.shape)
print("Columns:")
for c in out_df.columns:
    print("-", c)