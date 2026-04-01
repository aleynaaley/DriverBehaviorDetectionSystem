import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# AYARLAR
# =========================================================
INPUT_CSV = Path("/Users/aleyaley/Desktop/AB/dataset/data/features_temporal.csv")

OUTPUT_DIR = Path("/Users/aleyaley/Desktop/AB/model/sequence/class_weight/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 60
STRIDE = 5


TRAIN_PERSON_IDS = [2, 3, 4, 5, 6, 7, 9, 13, 15]
TEST_PERSON_IDS = [1, 11, 12, 14]

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
# KONTROL
# =========================================================
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Bulunamadı: {INPUT_CSV}")

# =========================================================
# VERİYİ OKU
# =========================================================
df = pd.read_csv(INPUT_CSV)

df = df.sort_values(
    ["person_id", "video_id", "segment_type", "frame_id"]
).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in DROP_COLS]

print("Toplam satır:", len(df))
print("Feature sayısı:", len(feature_cols))
print("İlk feature'lar:", feature_cols[:10])

# =========================================================
# SEQUENCE ÜRETİCİ
# =========================================================
def build_sequences(input_df: pd.DataFrame, person_ids: list[int], split_name: str):
    split_df = input_df[input_df["person_id"].isin(person_ids)].copy()
    split_df = split_df.sort_values(
        ["person_id", "video_id", "segment_type", "frame_id"]
    ).reset_index(drop=True)

    print(f"\n[{split_name}] satır sayısı:", len(split_df))
    print(f"[{split_name}] person_id dağılımı:", sorted(split_df['person_id'].unique().tolist()))
    print(f"[{split_name}] label dağılımı:")
    print(split_df["label"].value_counts())

    X_list = []
    y_list = []
    meta_list = []

    group_cols = ["person_id", "video_id", "segment_type"]

    for _, g in split_df.groupby(group_cols, sort=False):
        g = g.sort_values("frame_id").reset_index(drop=True)

        # Tek pencere içinde asla iki kişi karışmaz;
        # çünkü zaten person_id + video_id + segment_type bazında gruplanıyor.
        if len(g) < WINDOW_SIZE:
            continue

        X_g = g[feature_cols].values
        y_g = g["label"].values

        for start in range(0, len(g) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE

            seq_x = X_g[start:end]       # shape: (window, features)
            seq_y = y_g[end - 1]         # son frame label

            last_row = g.iloc[end - 1]

            X_list.append(seq_x)
            y_list.append(seq_y)

            meta_list.append({
                "person_id": int(last_row["person_id"]),
                "video_id": str(last_row["video_id"]),
                "segment_type": str(last_row["segment_type"]),
                "end_frame_id": int(last_row["frame_id"]),
                "end_timestamp_sec": float(last_row["timestamp_sec"]),
                "label": int(seq_y),
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_list)

    print(f"\n[{split_name}] sequence dataset oluşturuldu.")
    print(f"[{split_name}] X shape:", X.shape)
    print(f"[{split_name}] y shape:", y.shape)
    print(f"[{split_name}] meta shape:", meta_df.shape)

    if len(meta_df) > 0:
        print(f"[{split_name}] sequence label dağılımı:")
        print(meta_df["label"].value_counts())

    return X, y, meta_df

# =========================================================
# TRAIN / TEST AYRI ÜRET
# =========================================================
X_train, y_train, meta_train = build_sequences(df, TRAIN_PERSON_IDS, "TRAIN")
X_test, y_test, meta_test = build_sequences(df, TEST_PERSON_IDS, "TEST")

# =========================================================
# KAYDET
# =========================================================
np.save(OUTPUT_DIR / "X_train_sequences.npy", X_train)
np.save(OUTPUT_DIR / "y_train_sequences.npy", y_train)
meta_train.to_csv(OUTPUT_DIR / "train_sequence_meta.csv", index=False)

np.save(OUTPUT_DIR / "X_test_sequences.npy", X_test)
np.save(OUTPUT_DIR / "y_test_sequences.npy", y_test)
meta_test.to_csv(OUTPUT_DIR / "test_sequence_meta.csv", index=False)

with open(OUTPUT_DIR / "sequence_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "input_csv": str(INPUT_CSV),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "train_person_ids": TRAIN_PERSON_IDS,
        "test_person_ids": TEST_PERSON_IDS,
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }, f, indent=2, ensure_ascii=False)

print("\nKaydedilen dosyalar:")
print("-", OUTPUT_DIR / "X_train_sequences.npy")
print("-", OUTPUT_DIR / "y_train_sequences.npy")
print("-", OUTPUT_DIR / "train_sequence_meta.csv")
print("-", OUTPUT_DIR / "X_test_sequences.npy")
print("-", OUTPUT_DIR / "y_test_sequences.npy")
print("-", OUTPUT_DIR / "test_sequence_meta.csv")
print("-", OUTPUT_DIR / "sequence_config.json")