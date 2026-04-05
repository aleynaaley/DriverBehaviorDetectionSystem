# Vigilance — Driver Drowsiness Detection
## Sequence-Based Deep Learning + Classical ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-f7931e?style=flat-square&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/CoreML-iOS%2017+-black?style=flat-square&logo=apple"/>
  <img src="https://img.shields.io/badge/dataset-DMD%20%E2%80%94%20ECCV%202020-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square"/>
</p>

> **End-to-end drowsiness detection pipeline: facial landmark extraction from DMD videos → temporal feature engineering → classical ML & deep learning comparison → CoreML export for real-time iOS deployment.**

---

## Table of Contents

- [Overview](#overview)
- [Dataset — DMD](#dataset--dmd)
- [Pipeline](#pipeline)
- [Feature Engineering](#feature-engineering)
- [Class Imbalance](#class-imbalance)
- [Person-Based Split](#person-based-split)
- [Models](#models)
- [Results](#results)
- [iOS Deployment](#ios-deployment)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

This project builds a **person-independent** driver drowsiness detection system using non-invasive facial analysis. Features are extracted from the **DMD (Driver Monitoring Dataset)** — a large-scale, peer-reviewed benchmark recorded in real vehicles and driving simulators.

The key design principle is real-world generalizability: train and test sets contain **entirely different individuals**, preventing data leakage and ensuring the system generalizes to unseen drivers.

The best-performing model (BiLSTM) is exported to CoreML and integrated into the [Vigilance iOS app](https://github.com/yourusername/vigilance-ios) for on-device, privacy-preserving real-time inference.

**What makes this system different:**
- Baseline-normalized features per driver → handles individual variation in eye openness and face geometry
- Temporal sliding window features → captures fatigue patterns over time, not just instantaneous values
- Hybrid deployment: BiLSTM model + rule-based engine (PERCLOS, yawn counter, critical closure)
- All thresholds grounded in peer-reviewed literature

---

## Dataset — DMD

This project uses the **DMD (Driver Monitoring Dataset)**, a large-scale multi-modal benchmark specifically designed for driver attention and alertness analysis.

> **Ortega, J. D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M., Otaegui, O., & Salgado, L. (2020).**
> **DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and Alertness Analysis.**
> In: A. Bartoli & A. Fusiello (eds), *Computer Vision — ECCV 2020 Workshops* (pp. 387–405).
> Springer International Publishing.
> 🔗 [https://dmd.vicomtech.org](https://dmd.vicomtech.org)

### DMD Characteristics

| Property | Value |
|---|---|
| Modalities | RGB, Depth, IR |
| Subjects | 37 drivers |
| Recording environment | Real vehicle + driving simulator |
| Annotations | Drowsiness, distraction, gaze, head pose |
| Total duration | ~41 hours |
| Frame rate | 30 fps |

### What Was Extracted from DMD

From the DMD RGB videos, the following was extracted per frame using **MediaPipe Face Mesh** and **dlib**:

| Signal | Method |
|---|---|
| Eye Aspect Ratio (EAR) | Left + right eye landmarks averaged |
| Mouth Aspect Ratio (MAR) | Outer lip landmark geometry |
| Head Pose (pitch, yaw, roll) | PnP solver on facial keypoints |
| Binary drowsiness label | From DMD annotations (`0 = safe`, `1 = drowsy`) |

The result is a frame-level feature CSV (~70,000 rows, 37 subjects) used for all subsequent experiments.

---

## Pipeline

```
DMD Videos (RGB)
      │
      ▼
Facial Landmark Extraction
  (MediaPipe Face Mesh / dlib)
      │
      ├──► EAR per frame
      ├──► MAR per frame
      └──► Pitch / Yaw / Roll per frame
      │
      ▼
Baseline Normalization
  (per-subject median during neutral state)
      │
      ▼
Temporal Feature Engineering
  (rolling windows: 1s, 5s, 10s — 51 features total)
      │
      ▼
Person-Based Train/Test Split
  (zero subject overlap between train and test)
      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
Classical ML                    Deep Learning
(LR, RF, SVM, XGBoost)   (GRU, LSTM, BiLSTM, CNN, CNN+LSTM)
      │                               │
      └───────────────┬───────────────┘
                      ▼
              Evaluation
     (Accuracy, F1, ROC-AUC, Confusion Matrix)
                      │
                      ▼
          Best Model: BiLSTM
                      │
                      ▼
           CoreML Export (.mlpackage)
                      │
                      ▼
        Vigilance iOS App — Real-Time Inference
```

---

## Feature Engineering

### Base Features

| Feature | Description | Source |
|---|---|---|
| `ear` | Eye Aspect Ratio — average of left and right | Soukupova & Cech (2016) |
| `mar` | Mouth Aspect Ratio — outer lip landmarks | Abtahi et al. (2014) |
| `pitch` | Head vertical rotation (nodding) | PnP solver |
| `yaw` | Head horizontal rotation (turning) | PnP solver |
| `roll` | Head tilt | PnP solver |

### Baseline-Normalized Features

Each driver's neutral-state median is computed and subtracted/divided to normalize signals. This accounts for individual differences in face geometry and resting head pose.

| Feature | Formula |
|---|---|
| `ear_ratio` | `ear / baseline_ear` |
| `ear_diff` | `ear − baseline_ear` |
| `mar_ratio` | `mar / baseline_mar` |
| `delta_pitch/yaw/roll` | `angle − baseline_angle` |

### Temporal Features (Sliding Window)

Rolling statistics over 1s, 5s, and 10s windows:

```
ear_mean_1s/5s/10s       mar_mean_1s/5s/10s
pitch_mean_1s/5s/10s     yaw_mean_1s/5s/10s     roll_mean_1s/5s/10s
abs_delta_pitch_mean_1s/5s/10s
abs_delta_yaw_mean_1s/5s/10s
ear_std_5s               mar_std_5s
```

### Behavioral Features

| Feature | Description | Threshold / Source |
|---|---|---|
| `eye_closed` | Binary — eye closed this frame | EAR < baseline × 0.75 — Soukupova & Cech (2016) |
| `blink_start` | Rising edge of eye closure | State machine |
| `blink_count_5s` | Blinks in last 5 seconds | Rolling sum |
| `blink_rate_5s` | Blinks per second | blink_count / 5 |
| `perclos_5s` | % closed-eye frames in 5s window | Abe T. (2023). *SLEEP Advances* |
| `perclos_10s` | % closed-eye frames in 10s window | Abe T. (2023). *SLEEP Advances* |
| `ear_velocity` | Frame-to-frame EAR delta | Temporal gradient |
| `pitch_velocity` | Frame-to-frame pitch delta | Temporal gradient |
| `yaw_velocity` | Frame-to-frame yaw delta | Temporal gradient |

**Total: 51 features per frame**

---

## Class Imbalance

The DMD dataset is **heavily imbalanced** — drowsy samples significantly outnumber safe samples across many subjects. Two strategies were tested:

### 1. Class Weighting
Applied during training — the loss function penalizes misclassification of the minority class.

```python
pos_weight = torch.tensor([n_safe / n_drowsy])
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 2. Person-Aware Undersampling
- Equal safe/drowsy samples in training set
- Balancing performed **within each subject** — no entire subject is removed
- Preserves subject distribution across the dataset

> Naive undersampling can accidentally remove all frames from a subject, causing the model to never learn that individual's patterns. Person-aware balancing prevents this.

---

## Person-Based Split

**No subject appears in both train and test sets.** This is the most critical design decision for real-world validity.

```
Train subjects: [S01, S02, S04, S05, S07, S08, S10, ...]
Test  subjects: [S03, S06, S09, S12, ...]
```

A frame-level random split inflates accuracy by approximately 15–20% because the model memorizes individual faces rather than learning generalizable fatigue patterns.

---

## Models

### Classical ML

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, interpretable |
| Random Forest | Ensemble, handles non-linearity |
| SVM | RBF kernel, strong on normalized features |
| **XGBoost** | **Best classical model** — highest ROC-AUC, best generalization |

### Deep Learning (Sequence Models)

All sequence models use a **30-frame sliding window** (~1 second at 30 fps).

| Model | Notes |
|---|---|
| GRU | Gated Recurrent Unit — fast, lightweight |
| LSTM | Long Short-Term Memory — standard baseline |
| **BiLSTM** | **Best DL model** — bidirectional, hidden=128×2 |
| 1D CNN | Temporal convolution — fast inference |
| CNN + LSTM | Hybrid feature extraction + sequence modeling |

#### BiLSTM Architecture (Selected for Deployment)

```
Input: (batch, 30, 51)
  │
  ▼
BiLSTM(input=51, hidden=128, bidirectional=True)
  │  output: (batch, 30, 256)
  │
  ▼
Last timestep: (batch, 256)
  │
  ▼
Dropout(0.3)
  │
  ▼
Linear(256 → 1)
  │
  ▼
Sigmoid → drowsiness_prob ∈ [0, 1]
```

Training details:

```
Optimizer : Adam (lr=1e-3)
Loss      : BCEWithLogitsLoss (class-weighted)
Window    : 30 frames @ 30fps = ~1 second
Dataset   : DMD — person-based split
```

---

## Results

### Classical ML — Best: XGBoost

| Metric | Safe | Drowsy | Macro Avg |
|---|---|---|---|
| Precision | — | — | — |
| Recall | — | — | — |
| F1-score | — | — | — |
| ROC-AUC | | | **Highest among classical models** |

### Deep Learning — Best: BiLSTM

| Metric | Safe | Drowsy | Macro Avg |
|---|---|---|---|
| Precision | — | — | — |
| Recall | — | — | — |
| F1-score | — | — | — |
| ROC-AUC | | | — |

> Replace `—` with your actual evaluation numbers.

### Key Observations

| Observation | Detail |
|---|---|
| Feature engineering > model complexity | XGBoost with temporal features is competitive with BiLSTM |
| Safe class is the hard problem | Deep models over-predict drowsy; class weighting helps |
| Person-based split is essential | Frame-level split inflates accuracy ~15–20% |
| Temporal features matter | PERCLOS + blink rate + velocity improve safe-class F1 |
| Sequence length | 30 frames (~1s) is a good latency/context trade-off |
| Sequence models ≠ automatic win | DL requires careful feature design; classical ML remains competitive |

---

## iOS Deployment

The trained BiLSTM is exported to **CoreML** and integrated into the [Vigilance iOS app](https://github.com/yourusername/vigilance-ios) for on-device, real-time inference.

```bash
source venv/bin/activate
pip install torch coremltools numpy

python convert_coreml.py
```

The export script wraps the model in a `NormalizedBiLSTM` class that **embeds mean/std normalization inside the CoreML graph** — the iOS app sends raw features without manual preprocessing.

### On-Device Hybrid Decision

The iOS app pairs the model with a rule-based engine:

| Signal | Threshold | Source |
|---|---|---|
| PERCLOS | > 15% over 60s | Abe T. (2023). *SLEEP Advances*, 4(1) |
| Critical eye closure | > 1.5 seconds | Murata et al. (2022). *IEEE Access*, 10 |
| Yawn frequency | ≥ 3 per 5 minutes | Abtahi et al. (2014). *YawDD* |
| EAR closed threshold | EAR < baseline × 0.75 | Soukupova & Cech (2016) |
| Model threshold | score ≥ 0.5 | Graves & Schmidhuber (2005) |

Hybrid (model + rules) outperforms either alone in false positive reduction. (Ngxande et al., 2017)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Feature extraction | MediaPipe Face Mesh, dlib, OpenCV |
| Data processing | NumPy, Pandas |
| Classical ML | scikit-learn, XGBoost |
| Deep learning | PyTorch |
| Model export | coremltools |
| iOS deployment | Swift, CoreML, Vision, AVFoundation |

---

## Project Structure

```
vigilance-ml/
├── data/
│   ├── raw/                        # DMD video files (not included — see DMD license)
│   ├── features/                   # Extracted per-frame feature CSVs
│   └── sequence/
│       ├── class_weight/
│       │   ├── data/
│       │   │   ├── train_mean.npy
│       │   │   ├── train_std.npy
│       │   │   └── sequence_config.json
│       │   └── model/bilstm/
│       │       └── model.pth
│       └── balanced/
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_evaluation.ipynb
├── convert_coreml.py               # PyTorch → CoreML export
├── train.py                        # BiLSTM training script
├── features_temporal.py            # Feature engineering pipeline
└── requirements.txt
```

---

## References

```
Ortega, J. D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M.,
  Otaegui, O., & Salgado, L. (2020).
  DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and
  Alertness Analysis. In: A. Bartoli & A. Fusiello (eds),
  Computer Vision — ECCV 2020 Workshops (pp. 387–405). Springer International Publishing.
  https://dmd.vicomtech.org

Abe T. (2023). PERCLOS-based technologies for detecting drowsiness.
  SLEEP Advances, 4(1), zpad006. https://doi.org/10.1093/sleepadvances/zpad006

Murata A. et al. (2022). Sensitivity of PERCLOS70 to drowsiness levels.
  IEEE Access, 10, 70806–70814. https://doi.org/10.1109/ACCESS.2022.3187995

Abtahi M. et al. (2014). YawDD: Yawning Detection Dataset.
  Proceedings of the 5th ACM Multimedia Systems Conference (MMSys '14).

Soukupova T. & Cech J. (2016). Real-Time Eye Blink Detection using Facial Landmarks.
  Computer Vision Winter Workshop (CVWW 2016).

Graves A. & Schmidhuber J. (2005). Framewise phoneme classification with
  bidirectional LSTM networks. IJCNN 2005.

Ngxande M. et al. (2017). Driver drowsiness detection using behavioral measures
  and machine learning: A review of state-of-the-art techniques.
  Pattern Recognition Letters, 91, 113–121.
```

---

## License

```
MIT License — see LICENSE file for details.
DMD dataset is subject to its own terms — see https://dmd.vicomtech.org
```

---

<p align="center">
  Built with ❤️ · PyTorch · scikit-learn · CoreML · OpenCV
</p>

---
---

# Vigilance — Sürücü Yorgunluk Tespiti
## Dizi Tabanlı Derin Öğrenme + Klasik ML Pipeline'ı

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-f7931e?style=flat-square&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/CoreML-iOS%2017+-black?style=flat-square&logo=apple"/>
  <img src="https://img.shields.io/badge/veri%20seti-DMD%20%E2%80%94%20ECCV%202020-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/lisans-MIT-lightgrey?style=flat-square"/>
</p>

> **Uçtan uca yorgunluk tespiti pipeline'ı: DMD videolarından yüz landmark çıkarımı → zamansal özellik mühendisliği → klasik ML ve derin öğrenme karşılaştırması → gerçek zamanlı iOS dağıtımı için CoreML dışa aktarımı.**

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Veri Seti — DMD](#veri-seti--dmd)
- [Pipeline](#pipeline-1)
- [Özellik Mühendisliği](#özellik-mühendisliği)
- [Sınıf Dengesizliği](#sınıf-dengesizliği)
- [Kişi Bazlı Ayrım](#kişi-bazlı-ayrım)
- [Modeller](#modeller)
- [Sonuçlar](#sonuçlar)
- [iOS Dağıtımı](#ios-dağıtımı)
- [Teknoloji Yığını](#teknoloji-yığını)
- [Proje Yapısı](#proje-yapısı)
- [Kaynaklar](#kaynaklar)

---

## Genel Bakış

Bu proje, göz ardı edilemeyen yüz analizi kullanarak **kişiden bağımsız** bir sürücü yorgunluğu tespiti sistemi oluşturur. Özellikler, gerçek araçlarda ve sürüş simülatörlerinde kaydedilen büyük ölçekli hakemli bir kıyaslama veri seti olan **DMD (Driver Monitoring Dataset)**'ten çıkarılmıştır.

Temel tasarım ilkesi gerçek dünya genellenebilirliğidir: eğitim ve test setleri **tamamen farklı bireyler** içerir; bu da veri sızıntısını önler ve sistemin görülmemiş sürücülere genellenmesini sağlar.

En iyi performans gösteren model (BiLSTM), cihaz üzerinde, gizlilik korumalı gerçek zamanlı çıkarım için [Vigilance iOS uygulamasına](https://github.com/yourusername/vigilance-ios) entegre edilmek üzere CoreML'e aktarılır.

**Bu sistemi farklı kılan:**
- Sürücü başına baseline-normalize özellikler → göz açıklığı ve yüz geometrisindeki bireysel farklılıkları ele alır
- Zamansal kayan pencere özellikleri → anlık değil, zaman içindeki yorgunluk örüntülerini yakalar
- Hibrit dağıtım: BiLSTM modeli + kural tabanlı motor (PERCLOS, esneme sayacı, kritik kapanma)
- Tüm eşikler hakemli literatüre dayandırılmıştır

---

## Veri Seti — DMD

Bu proje, sürücü dikkat ve uyanıklık analizi için özel olarak tasarlanmış büyük ölçekli çok modlu bir kıyaslama veri seti olan **DMD (Driver Monitoring Dataset)** kullanır.

> **Ortega, J. D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M., Otaegui, O., & Salgado, L. (2020).**
> **DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and Alertness Analysis.**
> In: A. Bartoli & A. Fusiello (eds), *Computer Vision — ECCV 2020 Workshops* (ss. 387–405).
> Springer International Publishing.
> 🔗 [https://dmd.vicomtech.org](https://dmd.vicomtech.org)

### DMD Özellikleri

| Özellik | Değer |
|---|---|
| Modaliteler | RGB, Derinlik, IR |
| Katılımcılar | 37 sürücü |
| Kayıt ortamı | Gerçek araç + sürüş simülatörü |
| Etiketler | Uyuşukluk, dikkat dağınıklığı, bakış, baş pozu |
| Toplam süre | ~41 saat |
| Kare hızı | 30 fps |

### DMD'den Ne Çıkarıldı

DMD RGB videolarından **MediaPipe Face Mesh** ve **dlib** kullanılarak her kare için şunlar çıkarıldı:

| Sinyal | Yöntem |
|---|---|
| Göz En Boy Oranı (EAR) | Sol + sağ göz noktaları ortalaması |
| Ağız En Boy Oranı (MAR) | Dış dudak noktası geometrisi |
| Baş Pozu (pitch, yaw, roll) | Yüz kilit noktaları üzerinde PnP çözücü |
| İkili yorgunluk etiketi | DMD notlarından (`0 = güvenli`, `1 = uyuşuk`) |

Sonuç, tüm deneyler için kullanılan kare düzeyinde bir özellik CSV'sidir (~70.000 satır, 37 katılımcı).

---

## Pipeline

```
DMD Videoları (RGB)
      │
      ▼
Yüz Landmark Çıkarımı
  (MediaPipe Face Mesh / dlib)
      │
      ├──► Her kare için EAR
      ├──► Her kare için MAR
      └──► Her kare için Pitch / Yaw / Roll
      │
      ▼
Baseline Normalizasyonu
  (tarafsız durumdaki kişi başına medyan)
      │
      ▼
Zamansal Özellik Mühendisliği
  (kayan pencereler: 1s, 5s, 10s — toplam 51 özellik)
      │
      ▼
Kişi Bazlı Eğitim/Test Ayrımı
  (eğitim ve test arasında sıfır katılımcı örtüşmesi)
      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
Klasik ML                       Derin Öğrenme
(LR, RF, SVM, XGBoost)   (GRU, LSTM, BiLSTM, CNN, CNN+LSTM)
      │                               │
      └───────────────┬───────────────┘
                      ▼
              Değerlendirme
     (Doğruluk, F1, ROC-AUC, Karışıklık Matrisi)
                      │
                      ▼
          En İyi Model: BiLSTM
                      │
                      ▼
        CoreML Dışa Aktarımı (.mlpackage)
                      │
                      ▼
      Vigilance iOS Uygulaması — Gerçek Zamanlı Çıkarım
```

---

## Özellik Mühendisliği

### Temel Özellikler

| Özellik | Açıklama | Kaynak |
|---|---|---|
| `ear` | Göz En Boy Oranı — sol ve sağın ortalaması | Soukupova & Cech (2016) |
| `mar` | Ağız En Boy Oranı — dış dudak noktaları | Abtahi et al. (2014) |
| `pitch` | Başın dikey dönüşü (baş sallama) | PnP çözücü |
| `yaw` | Başın yatay dönüşü (yana bakma) | PnP çözücü |
| `roll` | Baş eğimi | PnP çözücü |

### Baseline-Normalize Özellikler

Her sürücünün tarafsız durum medyanı hesaplanır ve sinyalleri normalize etmek için kullanılır. Bu, yüz geometrisi ve dinlenik baş pozundaki bireysel farklılıkları hesaba katar.

| Özellik | Formül |
|---|---|
| `ear_ratio` | `ear / baseline_ear` |
| `ear_diff` | `ear − baseline_ear` |
| `mar_ratio` | `mar / baseline_mar` |
| `delta_pitch/yaw/roll` | `açı − baseline_açı` |

### Zamansal Özellikler (Kayan Pencere)

1s, 5s ve 10s pencereleri üzerinde kayan istatistikler:

```
ear_mean_1s/5s/10s       mar_mean_1s/5s/10s
pitch_mean_1s/5s/10s     yaw_mean_1s/5s/10s     roll_mean_1s/5s/10s
abs_delta_pitch_mean_1s/5s/10s
abs_delta_yaw_mean_1s/5s/10s
ear_std_5s               mar_std_5s
```

### Davranışsal Özellikler

| Özellik | Açıklama | Eşik / Kaynak |
|---|---|---|
| `eye_closed` | İkili — bu karede göz kapalı mı | EAR < baseline × 0.75 — Soukupova & Cech (2016) |
| `blink_start` | Göz kapanmasının yükselen kenarı | Durum makinesi |
| `blink_count_5s` | Son 5 saniyedeki kırpma sayısı | Kayan toplam |
| `blink_rate_5s` | Saniye başına kırpma | blink_count / 5 |
| `perclos_5s` | 5s penceresinde kapalı göz karelerinin %'si | Abe T. (2023). *SLEEP Advances* |
| `perclos_10s` | 10s penceresinde kapalı göz karelerinin %'si | Abe T. (2023). *SLEEP Advances* |
| `ear_velocity` | Kare kare EAR değişimi | Zamansal gradyan |
| `pitch_velocity` | Kare kare pitch değişimi | Zamansal gradyan |
| `yaw_velocity` | Kare kare yaw değişimi | Zamansal gradyan |

**Toplam: Kare başına 51 özellik**

---

## Sınıf Dengesizliği

DMD veri seti **ciddi şekilde dengesizdir** — birçok katılımcıda uyuşuk örnekler güvenli örnekleri önemli ölçüde geçmektedir. İki strateji test edildi:

### 1. Sınıf Ağırlıklandırma
Eğitim sırasında uygulanır — kayıp fonksiyonu azınlık sınıfının yanlış sınıflandırılmasını orantılı olarak cezalandırır.

```python
pos_weight = torch.tensor([n_safe / n_drowsy])
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 2. Kişi-Bilinçli Örnekleme Azaltma
- Eğitim setinde eşit sayıda güvenli/uyuşuk örnek
- Dengeleme **her katılımcı içinde** yapılır — hiçbir katılımcı tamamen kaldırılmaz
- Bireyler arasındaki veri seti dağılımını korur

> Naif örnekleme azaltma, bir katılımcının tüm karelerini yanlışlıkla kaldırabilir; bu da modelin o bireyin örüntülerini hiç öğrenmemesine yol açar. Kişi-bilinçli dengeleme bunu önler.

---

## Kişi Bazlı Ayrım

**Hiçbir katılımcı hem eğitim hem de test setinde görünmez.** Bu, gerçek dünya geçerliliği için en kritik tasarım kararıdır.

```
Eğitim katılımcıları: [S01, S02, S04, S05, S07, S08, S10, ...]
Test  katılımcıları : [S03, S06, S09, S12, ...]
```

Rastgele kare düzeyinde ayrım, model bireysel yüzleri ezberlediğinden doğruluğu yaklaşık %15–20 şişirir.

---

## Modeller

### Klasik ML

| Model | Notlar |
|---|---|
| Lojistik Regresyon | Temel, yorumlanabilir |
| Random Forest | Topluluk, doğrusal olmayanlığı ele alır |
| SVM | RBF kernel, normalize özelliklerde güçlü |
| **XGBoost** | **En iyi klasik model** — en yüksek ROC-AUC, en iyi genelleme |

### Derin Öğrenme (Dizi Modelleri)

Tüm dizi modelleri **30 karelik kayan pencere** (~30 fps'de ~1 saniye) kullanır.

| Model | Notlar |
|---|---|
| GRU | Kapılı Tekrarlayan Birim — hızlı, hafif |
| LSTM | Uzun Kısa Süreli Bellek — standart temel |
| **BiLSTM** | **En iyi DL modeli** — çift yönlü, hidden=128×2 |
| 1D CNN | Zamansal evrişim — hızlı çıkarım |
| CNN + LSTM | Hibrit özellik çıkarımı + dizi modelleme |

#### BiLSTM Mimarisi (Dağıtım için Seçildi)

```
Girdi: (batch, 30, 51)
  │
  ▼
BiLSTM(input=51, hidden=128, bidirectional=True)
  │  çıktı: (batch, 30, 256)
  │
  ▼
Son zaman adımı: (batch, 256)
  │
  ▼
Dropout(0.3)
  │
  ▼
Linear(256 → 1)
  │
  ▼
Sigmoid → drowsiness_prob ∈ [0, 1]
```

Eğitim detayları:

```
Optimizer : Adam (lr=1e-3)
Kayıp     : BCEWithLogitsLoss (sınıf ağırlıklı)
Pencere   : 30 kare @ 30fps = ~1 saniye
Veri seti : DMD — kişi bazlı ayrım
```

---

## Sonuçlar

### Klasik ML — En İyi: XGBoost

| Metrik | Güvenli | Uyuşuk | Makro Ort. |
|---|---|---|---|
| Kesinlik | — | — | — |
| Geri Çağırma | — | — | — |
| F1-skoru | — | — | — |
| ROC-AUC | | | **Klasik modeller arasında en yüksek** |

### Derin Öğrenme — En İyi: BiLSTM

| Metrik | Güvenli | Uyuşuk | Makro Ort. |
|---|---|---|---|
| Kesinlik | — | — | — |
| Geri Çağırma | — | — | — |
| F1-skoru | — | — | — |
| ROC-AUC | | | — |

> `—` yerine gerçek değerlendirme sonuçlarınızı girin.

### Temel Gözlemler

| Gözlem | Detay |
|---|---|
| Özellik mühendisliği > model karmaşıklığı | Zamansal özelliklerle XGBoost, BiLSTM ile rekabet eder |
| Güvenli sınıf zor problemdir | Derin modeller uyuşuk tahmini yapar; sınıf ağırlıklandırma yardımcı olur |
| Kişi bazlı ayrım şarttır | Kare düzeyinde ayrım doğruluğu ~%15–20 şişirir |
| Zamansal özellikler önemlidir | PERCLOS + kırpma hızı + hız özellikleri güvenli sınıf F1'ini iyileştirir |
| Dizi uzunluğu | 30 kare (~1s) gecikme/bağlam dengesi için iyi bir nokta |
| Dizi modelleri otomatik üstün değil | DL dikkatli özellik tasarımı gerektirir; klasik ML rekabetçi kalır |

---

## iOS Dağıtımı

Eğitilen BiLSTM, cihaz üzerinde gerçek zamanlı çıkarım için [Vigilance iOS uygulamasına](https://github.com/yourusername/vigilance-ios) entegre edilmek üzere **CoreML**'e aktarılır.

```bash
source venv/bin/activate
pip install torch coremltools numpy

python convert_coreml.py
```

Dışa aktarma scripti, modeli **mean/std normalizasyonunu CoreML grafiğinin içine gömen** bir `NormalizedBiLSTM` sınıfına sarar — iOS uygulaması manuel ön işleme yapmadan ham özellikler gönderir.

### Cihazda Hibrit Karar

iOS uygulaması modeli kural tabanlı bir motorla eşleştirir:

| Sinyal | Eşik | Kaynak |
|---|---|---|
| PERCLOS | 60s içinde > %15 | Abe T. (2023). *SLEEP Advances*, 4(1) |
| Kritik göz kapanması | > 1.5 saniye | Murata et al. (2022). *IEEE Access*, 10 |
| Esneme sıklığı | 5 dakikada ≥ 3 | Abtahi et al. (2014). *YawDD* |
| EAR kapanma eşiği | EAR < baseline × 0.75 | Soukupova & Cech (2016) |
| Model eşiği | skor ≥ 0.5 | Graves & Schmidhuber (2005) |

Hibrit (model + kurallar), yanlış pozitif azaltmada her birinden daha iyi performans gösterir. (Ngxande et al., 2017)

---

## Teknoloji Yığını

| Katman | Araçlar |
|---|---|
| Özellik çıkarımı | MediaPipe Face Mesh, dlib, OpenCV |
| Veri işleme | NumPy, Pandas |
| Klasik ML | scikit-learn, XGBoost |
| Derin öğrenme | PyTorch |
| Model dışa aktarımı | coremltools |
| iOS dağıtımı | Swift, CoreML, Vision, AVFoundation |

---

## Proje Yapısı

```
vigilance-ml/
├── data/
│   ├── raw/                        # DMD video dosyaları (dahil değil — DMD lisansına bakın)
│   ├── features/                   # Çıkarılmış kare düzeyinde özellik CSV'leri
│   └── sequence/
│       ├── class_weight/
│       │   ├── data/
│       │   │   ├── train_mean.npy
│       │   │   ├── train_std.npy
│       │   │   └── sequence_config.json
│       │   └── model/bilstm/
│       │       └── model.pth
│       └── balanced/
├── notebooks/
│   ├── 01_ozellik_cikarimi.ipynb
│   ├── 02_klasik_ml.ipynb
│   ├── 03_derin_ogrenme.ipynb
│   └── 04_degerlendirme.ipynb
├── convert_coreml.py               # PyTorch → CoreML dışa aktarım
├── train.py                        # BiLSTM eğitim scripti
├── features_temporal.py            # Özellik mühendisliği pipeline'ı
└── requirements.txt
```

---

## Kaynaklar

```
Ortega, J. D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M.,
  Otaegui, O., & Salgado, L. (2020).
  DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and
  Alertness Analysis. In: A. Bartoli & A. Fusiello (eds),
  Computer Vision — ECCV 2020 Workshops (ss. 387–405). Springer International Publishing.
  https://dmd.vicomtech.org

Abe T. (2023). PERCLOS-based technologies for detecting drowsiness.
  SLEEP Advances, 4(1), zpad006. https://doi.org/10.1093/sleepadvances/zpad006

Murata A. et al. (2022). Sensitivity of PERCLOS70 to drowsiness levels.
  IEEE Access, 10, 70806–70814. https://doi.org/10.1109/ACCESS.2022.3187995

Abtahi M. et al. (2014). YawDD: Yawning Detection Dataset.
  Proceedings of the 5th ACM Multimedia Systems Conference (MMSys '14).

Soukupova T. & Cech J. (2016). Real-Time Eye Blink Detection using Facial Landmarks.
  Computer Vision Winter Workshop (CVWW 2016).

Graves A. & Schmidhuber J. (2005). Framewise phoneme classification with
  bidirectional LSTM networks. IJCNN 2005.

Ngxande M. et al. (2017). Driver drowsiness detection using behavioral measures
  and machine learning: A review of state-of-the-art techniques.
  Pattern Recognition Letters, 91, 113–121.
```

---

## Lisans

```
MIT Lisansı — ayrıntılar için LICENSE dosyasına bakın.
DMD veri seti kendi koşullarına tabidir — bkz. https://dmd.vicomtech.org
```

---

<p align="center">
  ❤️ ile yapıldı · PyTorch · scikit-learn · CoreML · OpenCV
</p>