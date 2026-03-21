import csv
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# =========================================================
# AYARLAR
# =========================================================
DATASET_ROOT = Path("../data")
OUTPUT_CSV = DATASET_ROOT / "calibration_summary.csv"

# Şimdilik kullanılmayacak klasörler
SKIP_FOLDERS = {"face8", "face10"}

# -----------------------------
# Landmark sabitleri
# -----------------------------
# EAR için 6 noktalı klasik set
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Ağız iç dudak contour'u
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Yüz ovali
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Head pose için:
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

def find_calibration_video(face_dir: Path, face_num: int) -> Path:
    video_path = face_dir / f"calibration_{face_num}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"{video_path} bulunamadı.")
    return video_path

def calculate_ear(landmarks_px, indices):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    A = dist.euclidean(landmarks_px[indices[1]], landmarks_px[indices[5]])
    B = dist.euclidean(landmarks_px[indices[2]], landmarks_px[indices[4]])
    C = dist.euclidean(landmarks_px[indices[0]], landmarks_px[indices[3]])
    return (A + B) / (2.0 * C) if C != 0 else np.nan

def calculate_polygon_area(landmarks_norm, indices, width, height):
    """
    Normalize landmarkları piksel koordinatına çevirip polygon alanını hesaplar.
    """
    points = np.array(
        [[landmarks_norm[i].x * width, landmarks_norm[i].y * height] for i in indices],
        dtype=np.float32
    )
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_head_pose(landmarks_norm, img_w, img_h):
    """
    solvePnP ile pitch, yaw, roll hesaplar.
    """
    # 3D yüz model noktaları (yaklaşık referans model)
    model_points = np.array([
        (0.0, 0.0, 0.0),           # nose tip
        (0.0, -330.0, -65.0),      # chin
        (-225.0, 170.0, -135.0),   # left eye outer corner
        (225.0, 170.0, -135.0),    # right eye outer corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0)    # right mouth corner
    ], dtype="double")

    image_points = np.array([
        (landmarks_norm[HEAD_POSE_POINTS[0]].x * img_w, landmarks_norm[HEAD_POSE_POINTS[0]].y * img_h),  # nose
        (landmarks_norm[HEAD_POSE_POINTS[1]].x * img_w, landmarks_norm[HEAD_POSE_POINTS[1]].y * img_h),  # chin
        (landmarks_norm[HEAD_POSE_POINTS[2]].x * img_w, landmarks_norm[HEAD_POSE_POINTS[2]].y * img_h),  # left eye
        (landmarks_norm[HEAD_POSE_POINTS[3]].x * img_w, landmarks_norm[HEAD_POSE_POINTS[3]].y * img_h),  # right eye
        (landmarks_norm[HEAD_POSE_POINTS[4]].x * img_w, landmarks_norm[HEAD_POSE_POINTS[4]].y * img_h),  # left mouth
        (landmarks_norm[HEAD_POSE_POINTS[5]].x * img_w, landmarks_norm[HEAD_POSE_POINTS[5]].y * img_h),  # right mouth
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return np.nan, np.nan, np.nan

    rmat, _ = cv2.Rodrigues(rotation_vector)
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

# =========================================================
# ANA İŞLEM
# =========================================================
rows = []

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    for face_dir in sorted(DATASET_ROOT.iterdir()):
        if not face_dir.is_dir():
            continue

        face_name = face_dir.name

        if face_name in SKIP_FOLDERS:
            print(f"[SKIP] {face_name}")
            continue

        try:
            face_num = extract_face_number(face_name)
            calib_video = find_calibration_video(face_dir, face_num)
        except Exception as e:
            print(f"[WARN] {face_name}: {e}")
            continue

        print(f"[INFO] Calibration işleniyor: {face_name}")

        cap = cv2.VideoCapture(str(calib_video))
        if not cap.isOpened():
            print(f"[WARN] {face_name}: video açılamadı")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        ear_vals = []
        mar_vals = []
        pitch_vals = []
        yaw_vals = []
        roll_vals = []

        frame_count = 0
        valid_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            lm = results.multi_face_landmarks[0].landmark
            pts_px = np.array(
                [np.multiply([p.x, p.y], [w, h]).astype(int) for p in lm]
            )

            # EAR
            ear_left = calculate_ear(pts_px, LEFT_EYE)
            ear_right = calculate_ear(pts_px, RIGHT_EYE)
            ear = np.nanmean([ear_left, ear_right])

            # MAR benzeri ağız/yüz oranı
            mouth_area = calculate_polygon_area(lm, INNER_LIPS, w, h)
            face_area = calculate_polygon_area(lm, FACE_OVAL, w, h)

            if face_area <= 0:
                continue

            mar = mouth_area / face_area

            # Head pose
            pitch, yaw, roll = get_head_pose(lm, w, h)

            if np.isnan(ear) or np.isnan(mar) or np.isnan(pitch) or np.isnan(yaw) or np.isnan(roll):
                continue

            ear_vals.append(ear)
            mar_vals.append(mar)
            pitch_vals.append(pitch)
            yaw_vals.append(yaw)
            roll_vals.append(roll)
            valid_count += 1

        cap.release()

        if valid_count == 0:
            print(f"[WARN] {face_name}: geçerli calibration frame yok")
            continue

        rows.append({
            "person_id": face_num,
            "video_id": face_name,
            "fps": round(float(fps), 4),
            "calibration_frames_total": frame_count,
            "calibration_frames_valid": valid_count,
            "baseline_ear": float(np.median(ear_vals)),
            "baseline_mar": float(np.median(mar_vals)),
            "baseline_pitch": float(np.median(pitch_vals)),
            "baseline_yaw": float(np.median(yaw_vals)),
            "baseline_roll": float(np.median(roll_vals)),
        })

        print(f"[OK] {face_name} tamamlandı")

# =========================================================
# CSV YAZ
# =========================================================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "person_id",
            "video_id",
            "fps",
            "calibration_frames_total",
            "calibration_frames_valid",
            "baseline_ear",
            "baseline_mar",
            "baseline_pitch",
            "baseline_yaw",
            "baseline_roll",
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nBitti. Calibration summary oluşturuldu: {OUTPUT_CSV}")