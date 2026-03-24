import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# PATH
# =========================================================
DATA_DIR = Path("/Users/aleyaley/Desktop/AB/model/sequence/balancing/data")
OUTPUT_DIR = Path("/Users/aleyaley/Desktop/AB/model/sequence/balancing/data/drop_balanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD
# =========================================================
X_train = np.load(DATA_DIR / "X_train_sequences.npy")
y_train = np.load(DATA_DIR / "y_train_sequences.npy")
meta_train = pd.read_csv(DATA_DIR / "train_sequence_meta.csv")

X_test = np.load(DATA_DIR / "X_test_sequences.npy")
y_test = np.load(DATA_DIR / "y_test_sequences.npy")

print("Orijinal train shape:", X_train.shape)
print("Orijinal test shape :", X_test.shape)

assert len(X_train) == len(y_train) == len(meta_train), "Train X / y / meta boyutları uyuşmuyor!"

# =========================================================
# DAĞILIM
# =========================================================
safe_idx = np.where(y_train == 0)[0]
drowsy_idx = np.where(y_train == 1)[0]

n_safe = len(safe_idx)
n_drowsy = len(drowsy_idx)

print("\nOrijinal train dağılımı:")
print("safe:", n_safe)
print("drowsy:", n_drowsy)

# =========================================================
# SAFE'LERİ AYNEN TUT
# DROWSY'LERİ KİŞİ BAZINDA ORANTILI AZALT
# =========================================================
np.random.seed(42)

safe_keep_idx = safe_idx.copy()

drowsy_meta = meta_train.iloc[drowsy_idx].copy()
drowsy_meta["orig_index"] = drowsy_idx

# kişi bazında drowsy sayıları
person_counts = drowsy_meta["person_id"].value_counts().sort_index()

print("\nKişi bazında orijinal drowsy sequence sayıları:")
print(person_counts.sort_index())

# hedef toplam drowsy = toplam safe
target_total_drowsy = n_safe

# kişi bazında orantılı hedefler
ratio = target_total_drowsy / n_drowsy
raw_targets = person_counts * ratio

# önce floor al
target_counts = np.floor(raw_targets).astype(int)

# kalan adetleri largest remainder yöntemi ile dağıt
remaining = target_total_drowsy - target_counts.sum()

remainders = (raw_targets - target_counts).sort_values(ascending=False)

for person_id in remainders.index[:remaining]:
    target_counts.loc[person_id] += 1

print("\nKişi bazında hedef drowsy sequence sayıları:")
print(target_counts.sort_index())
print("\nToplam hedef drowsy:", int(target_counts.sum()))

# kişi bazında sample seç
selected_drowsy_indices = []

for person_id, target_count in target_counts.sort_index().items():
    person_rows = drowsy_meta[drowsy_meta["person_id"] == person_id]["orig_index"].to_numpy()

    if target_count > len(person_rows):
        raise ValueError(f"person_id={person_id} için hedef count mevcut sayıyı aşıyor.")

    sampled = np.random.choice(person_rows, size=target_count, replace=False)
    selected_drowsy_indices.extend(sampled.tolist())

selected_drowsy_indices = np.array(selected_drowsy_indices, dtype=int)

# =========================================================
# YENİ BALANCED TRAIN
# =========================================================
balanced_indices = np.concatenate([safe_keep_idx, selected_drowsy_indices])
np.random.shuffle(balanced_indices)

X_train_bal = X_train[balanced_indices]
y_train_bal = y_train[balanced_indices]
meta_train_bal = meta_train.iloc[balanced_indices].reset_index(drop=True)

print("\nBalanced train shape:", X_train_bal.shape)
print("\nBalanced train dağılımı:")
print("safe:", int((y_train_bal == 0).sum()))
print("drowsy:", int((y_train_bal == 1).sum()))

print("\nBalanced train kişi bazında drowsy sequence sayıları:")
print(
    meta_train_bal[meta_train_bal["label"] == 1]["person_id"]
    .value_counts()
    .sort_index()
)

# =========================================================
# KAYDET
# =========================================================
np.save(OUTPUT_DIR / "X_train_sequences_balanced.npy", X_train_bal)
np.save(OUTPUT_DIR / "y_train_sequences_balanced.npy", y_train_bal)
meta_train_bal.to_csv(OUTPUT_DIR / "train_sequence_meta_balanced.csv", index=False)

# test aynen kalır
np.save(OUTPUT_DIR / "X_test_sequences.npy", X_test)
np.save(OUTPUT_DIR / "y_test_sequences.npy", y_test)

print("\nKaydedildi:")
print("-", OUTPUT_DIR / "X_train_sequences_balanced.npy")
print("-", OUTPUT_DIR / "y_train_sequences_balanced.npy")
print("-", OUTPUT_DIR / "train_sequence_meta_balanced.csv")
print("-", OUTPUT_DIR / "X_test_sequences.npy")
print("-", OUTPUT_DIR / "y_test_sequences.npy")