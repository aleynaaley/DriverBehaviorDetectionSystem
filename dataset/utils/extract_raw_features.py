import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# AYARLAR
# =========================================================
DATASET_ROOT = Path("../data")
MODEL_PATH = Path("../../models/face_landmarker.task")
CALIBRATION_CSV = DATASET_ROOT / "calibration_summary.csv"
OUTPUT_CSV = DATASET_ROOT / "features_raw.csv"

SKIP_FOLDERS = {"face8", "face10"}

# =========================================================
# LANDMARK INDEXLER
# =========================================================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

INNER_LIPS = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

# [nose_tip, chin, left_eye_outer, right_eye_outer, left_mouth_corner, right_mouth_corner]
HEAD_POSE_POINTS = [1, 152, 33, 263, 61, 291]

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def extract_face_number(face_name: str) -> int:
    m = re.search(r"face(\d+)", face_name)
    if not m:
        raise ValueError(f"Face numarası çıkarılamadı: {face_name}")
    return int(m.group(1))

def calculate_ear(pts_px: np.ndarray, indices):
    A = dist.euclidean(pts_px[indices[1]], pts_px[indices[5]])
    B = dist.euclidean(pts_px[indices[2]], pts_px[indices[4]])
    C = dist.euclidean(pts_px[indices[0]], pts_px[indices[3]])
    return (A + B) / (2.0 * C) if C != 0 else np.nan

def polygon_area(landmarks_norm, indices, w, h):
    pts = np.array([[landmarks_norm[i].x * w, landmarks_norm[i].y * h] for i in indices], dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_head_pose(landmarks_norm, w, h):
    model_points = np.array([
        (0.0, 0.0, 0.0),           # nose tip
        (0.0, -330.0, -65.0),      # chin
        (-225.0, 170.0, -135.0),   # left eye outer
        (225.0, 170.0, -135.0),    # right eye outer
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0)    # right mouth corner
    ], dtype="double")

    image_points = np.array([
        (landmarks_norm[HEAD_POSE_POINTS[0]].x * w, landmarks_norm[HEAD_POSE_POINTS[0]].y * h),
        (landmarks_norm[HEAD_POSE_POINTS[1]].x * w, landmarks_norm[HEAD_POSE_POINTS[1]].y * h),
        (landmarks_norm[HEAD_POSE_POINTS[2]].x * w, landmarks_norm[HEAD_POSE_POINTS[2]].y * h),
        (landmarks_norm[HEAD_POSE_POINTS[3]].x * w, landmarks_norm[HEAD_POSE_POINTS[3]].y * h),
        (landmarks_norm[HEAD_POSE_POINTS[4]].x * w, landmarks_norm[HEAD_POSE_POINTS[4]].y * h),
        (landmarks_norm[HEAD_POSE_POINTS[5]].x * w, landmarks_norm[HEAD_POSE_POINTS[5]].y * h),
    ], dtype="double")

    camera_matrix = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return np.nan, np.nan, np.nan

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy >= 1e-6:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = 0.0

    return pitch, yaw, roll

def load_calibration_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Calibration CSV bulunamadı: {path}")
    df = pd.read_csv(path)
    calib_map = {}
    for _, row in df.iterrows():
        calib_map[int(row["person_id"])] = {
            "video_id": row["video_id"],
            "fps": float(row["fps"]),
            "baseline_ear": float(row["baseline_ear"]),
            "baseline_mar": float(row["baseline_mar"]),
            "baseline_pitch": float(row["baseline_pitch"]),
            "baseline_yaw": float(row["baseline_yaw"]),
            "baseline_roll": float(row["baseline_roll"]),
        }
    return calib_map

# =========================================================
# BAŞLANGIÇ KONTROLLERİ
# =========================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

calib_map = load_calibration_summary(CALIBRATION_CSV)

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

rows = []

# =========================================================
# ANA İŞLEM
# =========================================================
for face_dir in sorted(DATASET_ROOT.iterdir()):
    if not face_dir.is_dir():
        continue

    face_name = face_dir.name
    if face_name in SKIP_FOLDERS:
        print(f"[SKIP] {face_name}")
        continue

    try:
        person_id = extract_face_number(face_name)
    except Exception as e:
        print(f"[WARN] {face_name}: {e}")
        continue

    if person_id not in calib_map:
        print(f"[WARN] {face_name}: calibration_summary.csv içinde bulunamadı")
        continue

    base = calib_map[person_id]

    # Her segment bağımsız video gibi işlenecek
    for segment_type, label in [("safe", 0), ("drowsy", 1)]:
        video_path = face_dir / f"{segment_type}_{person_id}.mp4"
        if not video_path.exists():
            print(f"[WARN] {face_name}: {video_path.name} bulunamadı")
            continue

        print(f"[INFO] Processing: {face_name} / {segment_type}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] {face_name} / {segment_type}: video açılamadı")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = base["fps"] if base["fps"] > 0 else 30.0

        frame_idx = 0
        valid_count = 0

        # Her video için AYRI landmarker -> timestamp sıfırdan başlayabilir
        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if not result.face_landmarks:
                    frame_idx += 1
                    continue

                lm = result.face_landmarks[0]
                pts_px = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

                # EAR
                ear_left = calculate_ear(pts_px, LEFT_EYE)
                ear_right = calculate_ear(pts_px, RIGHT_EYE)
                ear = np.nanmean([ear_left, ear_right])

                # MAR (şimdilik calibration ile aynı tanım)
                face_area = polygon_area(lm, FACE_OVAL, w, h)
                if face_area <= 0:
                    frame_idx += 1
                    continue

                mouth_area = polygon_area(lm, INNER_LIPS, w, h)
                mar = mouth_area / face_area

                # Head pose
                pitch, yaw, roll = get_head_pose(lm, w, h)

                if np.isnan(ear) or np.isnan(mar) or np.isnan(pitch) or np.isnan(yaw) or np.isnan(roll):
                    frame_idx += 1
                    continue

                rows.append({
                    "person_id": person_id,
                    "video_id": face_name,
                    "segment_type": segment_type,
                    "frame_id": frame_idx,
                    "timestamp_sec": float(frame_idx / fps),
                    "fps": float(fps),
                    "label": label,

                    "ear": float(ear),
                    "mar": float(mar),
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll": float(roll),

                    "baseline_ear": base["baseline_ear"],
                    "baseline_mar": base["baseline_mar"],
                    "baseline_pitch": base["baseline_pitch"],
                    "baseline_yaw": base["baseline_yaw"],
                    "baseline_roll": base["baseline_roll"],
                })

                valid_count += 1
                frame_idx += 1

        cap.release()
        print(f"[OK] {face_name} / {segment_type}: {valid_count} geçerli frame")

# =========================================================
# KAYDET
# =========================================================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nBitti. Raw feature dataset oluşturuldu: {OUTPUT_CSV}")