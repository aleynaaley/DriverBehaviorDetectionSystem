import json
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# PATH
# =========================================================
INPUT_CSV = Path("/Users/aleyaley/Desktop/AB/dataset/data/features_temporal.csv")
OUTPUT_DIR = Path("/Users/aleyaley/Desktop/AB/model/sequence/balancing/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONFIG
# =========================================================
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

required_cols = {
    "person_id",
    "video_id",
    "segment_type",
    "frame_id",
    "timestamp_sec",
    "label",
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Eksik sütunlar var: {missing}")

df = df.sort_values(
    ["person_id", "video_id", "segment_type", "frame_id"]
).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in DROP_COLS]

print("Toplam satır:", len(df))
print("Feature sayısı:", len(feature_cols))
print("İlk feature'lar:", feature_cols[:10])

# =========================================================
# GLOBAL KONTROLLER
# =========================================================
all_persons = sorted(df["person_id"].unique().tolist())
print("\nDataset person_ids:", all_persons)

train_set = set(TRAIN_PERSON_IDS)
test_set = set(TEST_PERSON_IDS)

assert train_set.isdisjoint(test_set), "Train ve test person_id overlap var!"
assert train_set.union(test_set).issubset(set(all_persons)), "Train/test kişi listesinde dataset'te olmayan person_id var!"

# =========================================================
# SEQUENCE ÜRETİCİ
# =========================================================
def build_sequences(split_df: pd.DataFrame, split_name: str):
    X_list = []
    y_list = []
    meta_rows = []

    grouped = split_df.groupby(["person_id", "video_id", "segment_type"], sort=False)

    n_groups = 0
    n_sequences = 0

    for (person_id, video_id, segment_type), g in grouped:
        n_groups += 1

        # Sıralama garanti
        g = g.sort_values("frame_id").reset_index(drop=True)

        # Güvenlik kontrolleri
        assert g["person_id"].nunique() == 1, f"{split_name}: bir grupta birden fazla person_id var!"
        assert g["video_id"].nunique() == 1, f"{split_name}: bir grupta birden fazla video_id var!"
        assert g["segment_type"].nunique() == 1, f"{split_name}: bir grupta birden fazla segment_type var!"

        if len(g) < WINDOW_SIZE:
            continue

        # Feature ve label
        X_g = g[feature_cols].to_numpy(dtype=np.float32)
        y_g = g["label"].to_numpy(dtype=np.int64)

        # Frame sırası strictly nondecreasing mi?
        frame_ids = g["frame_id"].to_numpy()
        assert np.all(np.diff(frame_ids) >= 0), f"{split_name}: frame_id sırası bozuk!"

        # Sliding window
        for start in range(0, len(g) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            window_df = g.iloc[start:end]

            # Window içi güvenlik kontrolleri
            assert window_df["person_id"].nunique() == 1, f"{split_name}: bir window içinde birden fazla person_id var!"
            assert window_df["video_id"].nunique() == 1, f"{split_name}: bir window içinde birden fazla video_id var!"
            assert window_df["segment_type"].nunique() == 1, f"{split_name}: bir window içinde birden fazla segment_type var!"

            seq_x = X_g[start:end]
            seq_y = int(y_g[end - 1])  # son frame label

            last_row = window_df.iloc[-1]

            X_list.append(seq_x)
            y_list.append(seq_y)

            meta_rows.append({
                "person_id": int(last_row["person_id"]),
                "video_id": str(last_row["video_id"]),
                "segment_type": str(last_row["segment_type"]),
                "end_frame_id": int(last_row["frame_id"]),
                "end_timestamp_sec": float(last_row["timestamp_sec"]),
                "label": seq_y,
            })

            n_sequences += 1

    if len(X_list) == 0:
        raise RuntimeError(f"{split_name}: Hiç sequence üretilmedi. WINDOW_SIZE çok büyük olabilir.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)

    # Son güvenlik kontrolleri
    assert len(X) == len(y) == len(meta_df), f"{split_name}: X/y/meta boyutları eşleşmiyor!"
    assert meta_df["person_id"].isin(split_df["person_id"].unique()).all(), f"{split_name}: meta içinde split dışı kişi var!"

    print(f"\n[{split_name}] group sayısı:", n_groups)
    print(f"[{split_name}] sequence sayısı:", n_sequences)
    print(f"[{split_name}] X shape:", X.shape)
    print(f"[{split_name}] y shape:", y.shape)
    print(f"[{split_name}] label dağılımı:")
    print(meta_df["label"].value_counts())
    print(f"[{split_name}] person_id dağılımı:")
    print(sorted(meta_df['person_id'].unique().tolist()))

    return X, y, meta_df

# =========================================================
# SPLIT
# =========================================================
train_df = df[df["person_id"].isin(TRAIN_PERSON_IDS)].copy()
test_df = df[df["person_id"].isin(TEST_PERSON_IDS)].copy()

print("\nTrain raw satır sayısı:", len(train_df))
print("Test raw satır sayısı :", len(test_df))

print("\nTrain person_ids:", sorted(train_df["person_id"].unique().tolist()))
print("Test person_ids :", sorted(test_df["person_id"].unique().tolist()))

# Güvenlik
assert set(train_df["person_id"].unique()).issubset(train_set), "Train set içine test kişisi karışmış!"
assert set(test_df["person_id"].unique()).issubset(test_set), "Test set içine train kişisi karışmış!"
assert set(train_df["person_id"].unique()).isdisjoint(set(test_df["person_id"].unique())), "Train/test kişi overlap var!"

# =========================================================
# TRAIN / TEST SEQUENCE ÜRET
# =========================================================
X_train, y_train, meta_train = build_sequences(train_df, "TRAIN")
X_test, y_test, meta_test = build_sequences(test_df, "TEST")

# Son kontrol: train ve test meta kişileri ayrık mı?
assert set(meta_train["person_id"].unique()).isdisjoint(set(meta_test["person_id"].unique())), "Sequence seviyesinde train/test kişi overlap var!"

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
        "train_raw_rows": int(len(train_df)),
        "test_raw_rows": int(len(test_df)),
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