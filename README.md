# 🦺 PPE Detection System — Fixed & Production Ready

A YOLOv8-based Personal Protective Equipment (PPE) detection system that detects
**jackets (safety vests)**, **hats (hardhats)**, and their **absence** in real time.

---

## 🐛 Root Cause Analysis — What Was Wrong

### 🔴 Critical Bug 1: Class ID Mismatch (Primary Failure)
The original `data.yaml` declared `nc: 3` with classes `jacket=0, hat=1, shoes=2`,
but the Roboflow dataset actually contains **11 classes** (Hardhat, Mask, NO-Hardhat,
NO-Mask, NO-Safety Vest, Person, Safety Vest, Safety Cone, machinery, vehicle, other).

The model was therefore trained on a **completely wrong class mapping**, meaning:
- Class 0 was Safety **Hardhat**, not "jacket"
- Class 6 was Safety **Vest**, not "shoes"
- Classes 3–10 in labels were silently ignored, corrupting training
- The trained model had no idea what "jacket" or "shoes" meant

### 🔴 Critical Bug 2: No Shoes Class in Dataset
The dataset has no "shoes" annotations at all. The original code expected to detect
shoes but the data never contained them.

### 🔴 Critical Bug 3: Hardcoded Windows Paths
Every file used `C:\Users\Harsha\PPE Detector\...` — the app crashed immediately on
any other machine or OS.

### 🟠 Bug 4: Underpowered Training
- Only **10 epochs** in train-9 (far too few — needs 100+)
- Trained on **CPU only** (`device: cpu`) which severely limits batch size and speed
- No augmentation improvements for lighting robustness

### 🟠 Bug 5: Wrong Confidence Threshold
`conf=0.1` is far too low — causes massive false positives and "ghost" detections.
Fixed to `conf=0.35`.

### 🟡 Bug 6: Broken Absence Logic
The original code used `if len(detected) == 0: return "No items detected"` —
this fired constantly even when a person was clearly visible, if the wrong
class IDs caused no valid detections.

### 🟡 Bug 7: No Video Support, Broken Webcam Streaming
Original `cam.stream()` used outdated Gradio API. Video upload not implemented.

### 🟡 Bug 8: Basic/Broken UI
`index.html` was a plain HTML form with no CSS. The Gradio app had no styling,
no confidence display, no per-item PPE status cards.

---

## ✅ Fixes Applied

| Area | Fix |
|------|-----|
| `data.yaml` | Corrected to `nc=5`, proper class names matching dataset |
| `dataset/labels/` | All label files remapped (11→5 classes, dropped irrelevant classes) |
| `ppe_detector.py` | New inference engine with correct class logic + absence detection |
| `app.py` | Complete rewrite: dark theme UI, status cards, alerts, video tab |
| `train.py` | 100 epochs, better augmentation, GPU-aware, proper optimizer |
| `test_model.py` | Cross-platform, tests full pipeline |
| `validate_dataset.py` | Pre-training health check |
| `requirements.txt` | Pinned versions, all deps listed |

---

## 📁 Project Structure

```
PPE_Detector_Fixed/
├── app.py                  # Main Gradio UI (rewritten)
├── ppe_detector.py         # Detection engine (new)
├── train.py                # Training script (fixed)
├── test_model.py           # Test script (fixed)
├── validate_dataset.py     # Dataset health checker (new)
├── data.yaml               # Fixed: nc=5, correct class names
├── requirements.txt        # Fixed: all deps with versions
├── yolov8s.pt              # Base pretrained weights (download)
├── dataset/
│   ├── images/
│   │   ├── train/  (1132 images)
│   │   ├── val/    (143 images)
│   │   └── test/   (141 images)
│   └── labels/             # ← REMAPPED labels (fixed)
│       ├── train/  (1142 .txt files)
│       ├── val/    (143 .txt files)
│       └── test/   (141 .txt files)
└── runs/
    └── detect/
        └── ppe_train/
            └── weights/
                └── best.pt  # ← Use this after training
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Validate Dataset (Recommended)
```bash
python validate_dataset.py
```
Expected output: `✅ Dataset validation PASSED`

### 3. Train the Model
```bash
# With GPU (recommended):
python train.py

# On CPU (slow, ~8-12 hours for 100 epochs):
python train.py  # auto-detects CPU if no GPU

# Quick test run (5 epochs):
# Edit train.py: EPOCHS = 5
```

### 4. Run the App
```bash
python app.py
```
Opens at: http://localhost:7860

### 5. Test on Images
```bash
python test_model.py                              # test set samples
python test_model.py --image path/to/image.jpg   # single image
python test_model.py --conf 0.25                 # lower confidence
```

---

## 📊 Class Mapping (Fixed)

| Class ID | Name | Description | Detection Color |
|----------|------|-------------|-----------------|
| 0 | `jacket` | Safety vest present ✅ | 🟢 Green |
| 1 | `hat` | Hardhat present ✅ | 🔵 Blue |
| 2 | `no-jacket` | Explicitly not wearing jacket ❌ | 🔴 Red |
| 3 | `no-hat` | Explicitly not wearing hardhat ❌ | 🔴 Dark Red |
| 4 | `person` | Person detected (no PPE info) | ⚫ Grey |

### Original → Fixed Remapping
```
Original class 0 (Hardhat)        → new class 1 (hat)
Original class 2 (NO-Hardhat)     → new class 3 (no-hat)
Original class 4 (NO-Safety Vest) → new class 2 (no-jacket)
Original class 5 (Person)         → new class 4 (person)
Original class 6 (Safety Vest)    → new class 0 (jacket)
Original 1,3,7,8,9,10             → DROPPED (mask, cone, machinery, vehicle)
```

---

## 🎯 Alert Logic

| Condition | Alert | Severity |
|-----------|-------|----------|
| `no-jacket` detected | "Person is NOT wearing Jacket" | 🔴 Critical |
| `no-hat` detected | "Person is NOT wearing Hat" | 🔴 Critical |
| Both absent | "Person is NOT wearing Hat and Jacket!" | 🚨 Critical |
| Person visible, no hat seen | "Hat not visible (may be missing)" | ⚠️ Warning |
| Person visible, no jacket seen | "Jacket not visible (may be missing)" | ⚠️ Warning |
| All PPE present | "PPE Compliant — Jacket, Hat detected" | ✅ OK |
| No person in frame | "No person detected in frame" | ℹ️ Info |

---

## 📈 Recommended Training Configuration (GPU)

```yaml
epochs: 100          # Minimum for good accuracy
batch:  16           # Increase to 32 with 8GB+ VRAM
imgsz:  640
device: 0            # First GPU
optimizer: SGD
lr0:    0.01
augment: mosaic=1.0, mixup=0.1, hsv_v=0.4
```

Expected metrics after 100 epochs on this dataset:
- mAP50: ~0.75–0.85
- Precision: ~0.80–0.90
- Recall: ~0.70–0.85

---

## 💡 Accuracy Improvement Suggestions

1. **Add more data**: 1132 training images is modest — aim for 3000+
2. **Add shoes class**: Annotate feet/shoe regions separately
3. **Use YOLOv8m or YOLOv8l**: Larger models for better accuracy
4. **Hard negative mining**: Add images with no PPE to reduce false positives
5. **Night/low-light images**: Add synthetic dark augmentation
6. **Multi-angle coverage**: Ensure top-down, side-view, and far-field images
7. **Use YOLOv8 with tracking**: SORT/ByteTrack for video consistency
8. **TTA (Test-Time Augmentation)**: Enable with `augment=True` at inference

---

## ⚙️ Configuration

Edit `ppe_detector.py` to adjust:
```python
PPEDetector(
    conf=0.35,    # Lower = more detections (more false positives)
                  # Higher = fewer detections (more misses)
    iou=0.45,     # NMS overlap threshold
)
```
