from pathlib import Path
import pandas as pd

# =========================================================
# AYARLAR
# =========================================================
BASE_DIR = Path(__file__).resolve().parent              # .../model/balancing
SPLITS_DIR = BASE_DIR / "splits"
BALANCED_DATA_DIR = BASE_DIR / "balanced_data"

TRAIN_CSV = SPLITS_DIR / "train_subject_split.csv"
TEST_CSV = SPLITS_DIR / "test_subject_split.csv"

BALANCED_TRAIN_CSV = BALANCED_DATA_DIR / "train_subject_split_balanced.csv"
COPIED_TEST_CSV = BALANCED_DATA_DIR / "test_subject_split.csv"

RANDOM_STATE = 42

# =========================================================
# KLASÖR
# =========================================================
BALANCED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# VERİYİ OKU
# =========================================================
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Train dosyası bulunamadı: {TRAIN_CSV}")

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test dosyası bulunamadı: {TEST_CSV}")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Orijinal train shape:", train_df.shape)
print("Orijinal test shape :", test_df.shape)
print()

print("Orijinal train label dağılımı:")
print(train_df["label"].value_counts())
print()

# =========================================================
# SADECE TRAIN SETİNİ DENGELE
# Undersampling: çoğunluk sınıfını azalt
# =========================================================
safe_df = train_df[train_df["label"] == 0].copy()
drowsy_df = train_df[train_df["label"] == 1].copy()

n_safe = len(safe_df)
n_drowsy = len(drowsy_df)

minority_count = min(n_safe, n_drowsy)

safe_balanced = safe_df.sample(n=minority_count, random_state=RANDOM_STATE)
drowsy_balanced = drowsy_df.sample(n=minority_count, random_state=RANDOM_STATE)

balanced_train_df = pd.concat([safe_balanced, drowsy_balanced], ignore_index=True)
balanced_train_df = balanced_train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

# =========================================================
# KAYDET
# =========================================================
balanced_train_df.to_csv(BALANCED_TRAIN_CSV, index=False)
test_df.to_csv(COPIED_TEST_CSV, index=False)

print("Balanced train shape:", balanced_train_df.shape)
print("Copied test shape   :", test_df.shape)
print()

print("Balanced train label dağılımı:")
print(balanced_train_df["label"].value_counts())
print()

print("Kaydedilen dosyalar:")
print("-", BALANCED_TRAIN_CSV)
print("-", COPIED_TEST_CSV)