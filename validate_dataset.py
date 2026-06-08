"""
Dataset Validation Script (FIXED)
====================================
Verifies dataset integrity before training.
Handles both structures:
  - dataset/images/{split}/  (correct, after fix_dataset.py)
  - dataset/{split}/         (flat, before fix_dataset.py)

Run: python validate_dataset.py
"""

import glob
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR  = PROJECT_ROOT / "dataset"

CLASS_NAMES = {0: "jacket", 1: "hat", 2: "no-jacket", 3: "no-hat", 4: "person"}
NC = 5
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_image_dir(split: str) -> Path:
    """Auto-detect whether images are in dataset/images/{split} or dataset/{split}."""
    proper = DATASET_DIR / "images" / split
    flat   = DATASET_DIR / split

    if proper.exists() and any(f.suffix in EXTS for f in proper.iterdir() if f.is_file()):
        return proper
    if flat.exists() and any(f.suffix in EXTS for f in flat.iterdir() if f.is_file()):
        return flat
    return proper  # default (may be empty)


def validate_split(split: str) -> dict:
    img_dir   = find_image_dir(split)
    label_dir = DATASET_DIR / "labels" / split

    images = set(p.stem for p in img_dir.glob("*") if p.suffix in EXTS) if img_dir.exists() else set()
    labels = set(p.stem for p in label_dir.glob("*.txt")) if label_dir.exists() else set()

    issues = []
    class_counts = Counter()
    empty_labels = 0
    bad_coords   = 0
    bad_cls      = 0

    # Path structure warning
    flat = DATASET_DIR / split
    proper = DATASET_DIR / "images" / split
    if flat.exists() and images and not proper.exists():
        issues.append(f"  ⚠️  Images are in dataset/{split}/ (flat). Run python fix_dataset.py first.")

    # Image/label mismatches
    no_label = images - labels
    if no_label:
        issues.append(f"  ⚠️  {len(no_label)} images missing label files")

    orphan_labels = labels - images
    if orphan_labels:
        issues.append(f"  ⚠️  {len(orphan_labels)} label files without images — run fix_dataset.py")

    if not images:
        issues.append(f"  ❌ No images found in {img_dir}")

    # Validate each label file
    if label_dir.exists():
        for label_file in label_dir.glob("*.txt"):
            text = label_file.read_text().strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            if not lines:
                empty_labels += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"  ❌ Bad format in {label_file.name}: '{line[:40]}'")
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except ValueError:
                    issues.append(f"  ❌ Non-numeric values in {label_file.name}")
                    continue

                if cls_id < 0 or cls_id >= NC:
                    bad_cls += 1
                    if bad_cls <= 5:
                        issues.append(f"  ❌ Invalid class {cls_id} in {label_file.name} (expected 0-{NC-1})")
                else:
                    class_counts[cls_id] += 1

                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    bad_coords += 1
                    if bad_coords <= 3:
                        issues.append(f"  ❌ Out-of-range coords in {label_file.name}")

    return {
        "images": len(images), "labels": len(labels),
        "img_dir": img_dir,
        "class_counts": class_counts,
        "empty_labels": empty_labels,
        "bad_cls": bad_cls,
        "bad_coords": bad_coords,
        "issues": issues,
    }


def main():
    print("=" * 60)
    print("  PPE Dataset Validation")
    print("=" * 60)

    overall_ok = True

    for split in ["train", "val", "test"]:
        print(f"\n📂 {split.upper()} SET")
        result = validate_split(split)

        print(f"  Images: {result['images']}  (from {result['img_dir'].relative_to(PROJECT_ROOT)})")
        print(f"  Labels: {result['labels']}")
        print(f"  Empty label files: {result['empty_labels']}")

        print(f"\n  Class distribution:")
        total = sum(result['class_counts'].values())
        for cls_id in sorted(CLASS_NAMES.keys()):
            cnt = result['class_counts'].get(cls_id, 0)
            bar = "█" * int(cnt / max(total, 1) * 30)
            print(f"    {cls_id} {CLASS_NAMES[cls_id]:12s}: {cnt:5d}  {bar}")

        if result['bad_cls']:
            print(f"\n  ❌ {result['bad_cls']} annotations with invalid class IDs!")
            overall_ok = False
        if result['bad_coords']:
            print(f"\n  ❌ {result['bad_coords']} annotations with bad coordinates!")
            overall_ok = False

        if result['issues']:
            print(f"\n  Issues found:")
            for issue in result['issues'][:10]:
                print(f"  {issue}")
            if len(result['issues']) > 10:
                print(f"  ... and {len(result['issues'])-10} more")
            overall_ok = False
        else:
            print(f"\n  ✅ No issues found in {split} set")

    print("\n" + "=" * 60)
    if overall_ok:
        print("  ✅ Dataset validation PASSED — Ready to train!")
        print("\n  Next step:  python train.py")
    else:
        print("  ❌ Dataset validation FAILED — Fix issues before training")
        print("\n  Fix steps:")
        print("    1. python fix_dataset.py   ← moves images to correct path")
        print("    2. python validate_dataset.py   ← re-run to confirm")
    print("=" * 60)


if __name__ == "__main__":
    main()
