import json
import re
import csv
import subprocess
from pathlib import Path

# =========================================================
# AYARLAR
# =========================================================
DATASET_ROOT = Path("../data")
MASTER_CSV_PATH = DATASET_ROOT / "segments_master.csv"

# Calibration: başlangıç + 5 sn
CALIB_STARTS = {
    "face1": 1,
    "face2": 1,
    "face3": 1,
    "face4": 4,
    "face5": 5,
    "face6": 4,
    "face7": 5,
    "face9": 6,
    "face11": 12,
    "face12": 3,
    "face13": 17,
    "face14": 6,
    "face15": 10,
}
CALIB_DURATION_SEC = 5

# Safe aralıkları (safe tablosu)
SAFE_RANGES = {
    "face1":  (0, 30),
    "face2":  (0, 33),
    "face3":  (0, 30),
    "face4":  (0, 30),
    "face5":  (0, 24),
    "face6":  (0, 46),
    "face7":  (0, 24),
    "face9":  (0, 30),
    "face10": (0, 30),
    "face11": (0, 41),
    "face12": (0, 40),
    "face13": (0, 43),
    "face14": (0, 40),
    "face15": (0, 48),
}

# face8 ve face10 için özel durum
SKIP_FOLDERS = {"face8", "face10"}  # şimdilik ikisini dışarıda bırakıyoruz

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)

def ffprobe_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    out = subprocess.check_output(cmd).decode().strip()
    num, den = out.split("/")
    return float(num) / float(den)

def ffmpeg_cut(input_video: Path, output_video: Path, start_sec: float, end_sec: float):
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        print(f"[WARN] Geçersiz süre: {output_video.name} ({start_sec}-{end_sec})")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-ss", str(start_sec),
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-movflags", "+faststart",
        "-an",
        str(output_video)
    ]
    subprocess.run(cmd, check=True)
    return True

def sec_to_frame(sec: float, fps: float) -> int:
    return int(round(sec * fps))

def find_rgb_face_video(face_dir: Path):
    mp4s = list(face_dir.glob("*rgb_face.mp4"))
    if len(mp4s) != 1:
        raise RuntimeError(f"{face_dir} içinde 1 adet rgb_face.mp4 bulunmalı.")
    return mp4s[0]

def find_json_if_exists(face_dir: Path):
    jsons = list(face_dir.glob("*rgb_ann_drowsiness.json"))
    if len(jsons) == 1:
        return jsons[0]
    return None

def extract_face_number(face_name: str):
    m = re.search(r"face(\d+)", face_name)
    if not m:
        raise ValueError(f"Face numarası çıkarılamadı: {face_name}")
    return m.group(1)

# =========================================================
# ANA İŞLEM
# =========================================================
rows = []

for face_dir in sorted(DATASET_ROOT.iterdir()):
    if not face_dir.is_dir():
        continue

    face_name = face_dir.name

    if face_name in SKIP_FOLDERS:
        print(f"[SKIP] {face_name}")
        continue

    if face_name not in CALIB_STARTS or face_name not in SAFE_RANGES:
        print(f"[WARN] Metadata eksik: {face_name}")
        continue

    try:
        video_path = find_rgb_face_video(face_dir)
        json_path = find_json_if_exists(face_dir)
    except Exception as e:
        print(f"[ERROR] {face_name}: {e}")
        continue

    print(f"[INFO] İşleniyor: {face_name}")

    face_num = extract_face_number(face_name)
    fps = ffprobe_fps(video_path)
    duration = ffprobe_duration(video_path)

    # Calibration
    calib_start = CALIB_STARTS[face_name]
    calib_end = calib_start + CALIB_DURATION_SEC

    # Safe
    safe_start, safe_end = SAFE_RANGES[face_name]

    # Drowsy
    drowsy_start = safe_end
    drowsy_end = duration

    # Frame bilgileri
    calib_start_frame = sec_to_frame(calib_start, fps)
    calib_end_frame = sec_to_frame(calib_end, fps)

    safe_start_frame = sec_to_frame(safe_start, fps)
    safe_end_frame = sec_to_frame(safe_end, fps)

    drowsy_start_frame = sec_to_frame(drowsy_start, fps)
    drowsy_end_frame = sec_to_frame(drowsy_end, fps)

    # Çıktı dosyaları
    calib_out = face_dir / f"calibration_{face_num}.mp4"
    safe_out = face_dir / f"safe_{face_num}.mp4"
    drowsy_out = face_dir / f"drowsy_{face_num}.mp4"
    metadata_out = face_dir / f"metadata_{face_num}.json"

    # Videoları kes
    ffmpeg_cut(video_path, calib_out, calib_start, calib_end)
    ffmpeg_cut(video_path, safe_out, safe_start, safe_end)
    ffmpeg_cut(video_path, drowsy_out, drowsy_start, drowsy_end)

    metadata = {
        "face_folder": face_name,
        "face_number": int(face_num),
        "original_video": video_path.name,
        "annotation_json": json_path.name if json_path else None,
        "fps": fps,
        "video_duration_sec": duration,
        "calibration": {
            "start_sec": calib_start,
            "end_sec": calib_end,
            "start_frame": calib_start_frame,
            "end_frame": calib_end_frame,
            "output_file": calib_out.name
        },
        "safe": {
            "start_sec": safe_start,
            "end_sec": safe_end,
            "start_frame": safe_start_frame,
            "end_frame": safe_end_frame,
            "output_file": safe_out.name
        },
        "drowsy": {
            "start_sec": drowsy_start,
            "end_sec": drowsy_end,
            "start_frame": drowsy_start_frame,
            "end_frame": drowsy_end_frame,
            "output_file": drowsy_out.name
        }
    }

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    rows.append({
        "face_folder": face_name,
        "face_number": face_num,
        "original_video": video_path.name,
        "annotation_json": json_path.name if json_path else "",
        "fps": f"{fps:.4f}",
        "video_duration_sec": f"{duration:.2f}",
        "calib_start_sec": calib_start,
        "calib_end_sec": calib_end,
        "safe_start_sec": safe_start,
        "safe_end_sec": safe_end,
        "drowsy_start_sec": drowsy_start,
        "drowsy_end_sec": f"{drowsy_end:.2f}",
        "calibration_file": calib_out.name,
        "safe_file": safe_out.name,
        "drowsy_file": drowsy_out.name,
    })

    print(f"[OK] {face_name} tamamlandı")

# Master CSV
with open(MASTER_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "face_folder", "face_number", "original_video", "annotation_json",
            "fps", "video_duration_sec",
            "calib_start_sec", "calib_end_sec",
            "safe_start_sec", "safe_end_sec",
            "drowsy_start_sec", "drowsy_end_sec",
            "calibration_file", "safe_file", "drowsy_file"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nTüm işlem bitti. Master CSV oluşturuldu: {MASTER_CSV_PATH}")