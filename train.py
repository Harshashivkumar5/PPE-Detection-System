"""
PPE Detection - Training Script (FIXED)
========================================
Fixes applied:
1. Uses corrected data.yaml with 5 classes (not broken nc=3 with 11-class data)
2. Proper epochs (100) - original was only 10-50 on CPU which is insufficient
3. Better augmentation settings for PPE detection
4. Class-aware training with proper imgsz
5. Cross-platform path handling (no Windows absolute paths)
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_YAML    = str(PROJECT_ROOT / "data.yaml")
PRETRAINED   = str(PROJECT_ROOT / "yolov8s.pt")   # YOLOv8s for better accuracy
EPOCHS       = 100
IMGSZ        = 640
BATCH        = 16   # Reduce to 8 if OOM on GPU, or 4 on CPU
PATIENCE     = 30   # Early stopping
WORKERS      = 4

# Detect GPU availability
import torch
device = "0" if torch.cuda.is_available() else "cpu"
print(f"Training on: {'GPU (CUDA)' if device == '0' else 'CPU'}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH}, Image size: {IMGSZ}")

# ─── MODEL ────────────────────────────────────────────────────────────────────
# Download pretrained base model if not present
if not Path(PRETRAINED).exists():
    print(f"Downloading {PRETRAINED}...")
    model = YOLO("yolov8s.pt")  # auto-downloads
else:
    model = YOLO(PRETRAINED)

# ─── TRAINING ─────────────────────────────────────────────────────────────────
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    device=device,
    workers=WORKERS,
    project=str(PROJECT_ROOT / "runs" / "detect"),
    name="ppe_train",
    exist_ok=False,

    # Optimizer
    optimizer="SGD",
    lr0=0.01,
    lrf=0.001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,

    # Loss weights - upweight classification for PPE
    cls=0.8,   # Higher cls weight for better class discrimination
    box=7.5,
    dfl=1.5,

    # Augmentation - critical for PPE robustness
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,       # Handle low-light conditions
    degrees=10.0,    # Slight rotation
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,      # Mosaic augmentation
    mixup=0.1,       # MixUp for better generalization
    copy_paste=0.1,  # Copy-paste for partial visibility

    # Validation
    val=True,
    plots=True,
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs
    verbose=True,
)

print("\n" + "="*60)
print("✅ Training Complete!")
print(f"Best model: {PROJECT_ROOT}/runs/detect/ppe_train/weights/best.pt")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print("="*60)
