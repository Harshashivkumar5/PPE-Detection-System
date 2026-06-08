"""
fix_dataset.py — Run this ONCE after extracting the ZIP.

What it does:
  1. Moves images from flat dataset/train|val|test/ → dataset/images/train|val|test/
  2. Removes 10 duplicate/orphan label files (image940(1).txt etc.)
  3. Verifies everything is correct

Usage:
    python fix_dataset.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"

EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def move_flat_images():
    moved = 0
    for split in ["train", "val", "test"]:
        src_dir = DATASET / split          # flat: dataset/train/
        dst_dir = DATASET / "images" / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"  ⏭  No flat {split}/ dir found — skipping")
            continue

        files = [f for f in src_dir.iterdir() if f.suffix in EXTS]
        if not files:
            print(f"  ⏭  {split}/ is empty or has no images")
            continue

        print(f"  Moving {len(files)} images: dataset/{split}/ → dataset/images/{split}/")
        for f in files:
            shutil.move(str(f), str(dst_dir / f.name))
            moved += 1

        # Remove now-empty flat dir
        try:
            src_dir.rmdir()
        except OSError:
            pass  # not empty — leave it

    return moved


def remove_orphan_labels():
    removed = 0
    for split in ["train", "val", "test"]:
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_stems = {p.stem for p in img_dir.iterdir() if p.suffix in EXTS}
        for lbl in lbl_dir.glob("*.txt"):
            if lbl.stem not in img_stems:
                print(f"  Removing orphan label: {lbl.name}")
                lbl.unlink()
                removed += 1

    return removed


def count(split):
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    imgs = len([f for f in img_dir.iterdir() if f.suffix in EXTS]) if img_dir.exists() else 0
    lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    return imgs, lbls


def main():
    print("=" * 60)
    print("  PPE Dataset Fix Script")
    print("=" * 60)

    # Step 1: move flat images
    print("\n[1/3] Moving flat images to dataset/images/{split}/...")
    moved = move_flat_images()
    if moved == 0:
        print("  ✅ Images already in correct location")
    else:
        print(f"  ✅ Moved {moved} images")

    # Step 2: remove orphan labels
    print("\n[2/3] Removing orphan/duplicate label files...")
    removed = remove_orphan_labels()
    if removed == 0:
        print("  ✅ No orphan labels found")
    else:
        print(f"  ✅ Removed {removed} orphan labels")

    # Step 3: verify
    print("\n[3/3] Final counts:")
    ok = True
    for split in ["train", "val", "test"]:
        imgs, lbls = count(split)
        status = "✅" if imgs == lbls and imgs > 0 else "❌"
        print(f"  {status} {split:6s}: {imgs} images, {lbls} labels")
        if imgs == 0 or imgs != lbls:
            ok = False

    print("\n" + "=" * 60)
    if ok:
        print("  ✅ Dataset is ready! Run: python validate_dataset.py")
        print("  Then run:            python train.py")
    else:
        print("  ❌ Issues remain. Check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
