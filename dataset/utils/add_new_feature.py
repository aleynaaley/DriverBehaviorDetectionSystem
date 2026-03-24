import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path("/Users/aleyaley/Desktop/AB/dataset/data/features_temporal.csv")
OUTPUT_CSV = Path("/Users/aleyaley/Desktop/AB/dataset/data/extra/features_temporal_extra.csv")

EAR_THRESHOLD = 0.21
LONG_CLOSURE_FRAMES = 15
BLINK_WINDOW = 30

df = pd.read_csv(INPUT_CSV)

df = df.sort_values(["person_id", "video_id", "segment_type", "frame_id"]).reset_index(drop=True)

all_groups = []

for _, g in df.groupby(["person_id", "video_id", "segment_type"], sort=False):
    g = g.sort_values("frame_id").copy()

    # 1) eye_closed
    g["eye_closed"] = (g["ear"] < EAR_THRESHOLD).astype(int)

    # 2) eye_closed_duration
    durations = []
    counter = 0
    for val in g["eye_closed"]:
        if val == 1:
            counter += 1
        else:
            counter = 0
        durations.append(counter)
    g["eye_closed_duration"] = durations

    # 3) long_closure_flag
    g["long_closure_flag"] = (g["eye_closed_duration"] >= LONG_CLOSURE_FRAMES).astype(int)

    # 4) blink_flag (closed -> open geçişi)
    eye_vals = g["eye_closed"].to_numpy()
    blink_flag = np.zeros(len(g), dtype=int)
    for i in range(1, len(g)):
        if eye_vals[i - 1] == 1 and eye_vals[i] == 0:
            blink_flag[i] = 1
    g["blink_flag"] = blink_flag

    # 5) blink_count_window
    g["blink_count_window"] = g["blink_flag"].rolling(window=BLINK_WINDOW, min_periods=1).sum()

    all_groups.append(g)

df_out = pd.concat(all_groups, ignore_index=True)

df_out.to_csv(OUTPUT_CSV, index=False)

print("Bitti:", OUTPUT_CSV)
print("Shape:", df_out.shape)
print("Son eklenen sütunlar:")
print([
    "eye_closed",
    "eye_closed_duration",
    "long_closure_flag",
    "blink_flag",
    "blink_count_window",
])

print("\nKolonlar:")
for c in df_out.columns:
    print("-", c)