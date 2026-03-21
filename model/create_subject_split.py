import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# =========================================================
# AYARLAR
# =========================================================
BASE_DIR = Path(__file__).resolve().parent      # dataset/model
DATASET_DIR = BASE_DIR.parent                   # dataset
DATA_DIR = DATASET_DIR / "dataset/data"                 # dataset/data

CLASS_WEIGHT_DIR = BASE_DIR / "class_weight"    # dataset/model/class_weight
BALANCING_DIR = BASE_DIR / "balancing"          # dataset/model/balancing

FEATURES_DATA = DATA_DIR / "features_temporal.csv"

TEST_SIZE = 0.30
RANDOM_STATE = 42

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def make_subject_split(df: pd.DataFrame):
    person_ids = sorted(df["person_id"].unique().tolist())

    train_persons, test_persons = train_test_split(
        person_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_persons = sorted(train_persons)
    test_persons = sorted(test_persons)

    train_df = df[df["person_id"].isin(train_persons)].copy()
    test_df = df[df["person_id"].isin(test_persons)].copy()

    return train_persons, test_persons, train_df, test_df

# =========================================================
# ANA İŞLEM
# =========================================================
if not FEATURES_DATA.exists():
    raise FileNotFoundError(f"Bulunamadı: {FEATURES_DATA}")

df = pd.read_csv(FEATURES_DATA)

required_cols = {"person_id", "label"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Eksik sütunlar: {missing}")

train_persons, test_persons, train_df, test_df = make_subject_split(df)

# =========================================================
# CLASS_WEIGHT KAYITLARI
# =========================================================
cw_splits_dir = CLASS_WEIGHT_DIR / "splits"
ensure_dir(cw_splits_dir)

save_json(train_persons, cw_splits_dir / "train_person_ids.json")
save_json(test_persons, cw_splits_dir / "test_person_ids.json")

train_df.to_csv(cw_splits_dir / "train_subject_split.csv", index=False)
test_df.to_csv(cw_splits_dir / "test_subject_split.csv", index=False)

# =========================================================
# BALANCING KAYITLARI
# Aynı split buraya da kopyalanıyor
# =========================================================
bal_splits_dir = BALANCING_DIR / "splits"
ensure_dir(bal_splits_dir)

save_json(train_persons, bal_splits_dir / "train_person_ids.json")
save_json(test_persons, bal_splits_dir / "test_person_ids.json")

train_df.to_csv(bal_splits_dir / "train_subject_split.csv", index=False)
test_df.to_csv(bal_splits_dir / "test_subject_split.csv", index=False)

# =========================================================
# RAPOR
# =========================================================
print("Split tamamlandı.\n")

print("Train person_ids:", train_persons)
print("Test person_ids :", test_persons)
print()

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print()

print("Train label dağılımı:")
print(train_df["label"].value_counts())
print()

print("Test label dağılımı:")
print(test_df["label"].value_counts())
print()

print("Kaydedilen dosyalar:")
print("-", cw_splits_dir / "train_person_ids.json")
print("-", cw_splits_dir / "test_person_ids.json")
print("-", cw_splits_dir / "train_subject_split.csv")
print("-", cw_splits_dir / "test_subject_split.csv")
print("-", bal_splits_dir / "train_person_ids.json")
print("-", bal_splits_dir / "test_person_ids.json")
print("-", bal_splits_dir / "train_subject_split.csv")
print("-", bal_splits_dir / "test_subject_split.csv")