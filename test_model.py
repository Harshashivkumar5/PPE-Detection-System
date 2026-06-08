"""
PPE Model Test Script (FIXED)
================================
Tests the trained model on sample images from the test set.
Cross-platform - no hardcoded Windows paths.

Usage:
    python test_model.py                    # tests on dataset/images/test/
    python test_model.py --image path.jpg   # test single image
    python test_model.py --conf 0.3         # custom confidence
"""

import argparse
import cv2
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppe_detector import PPEDetector

def test_single(detector, image_path: Path, save_dir: Path):
    """Test on one image and show/save result."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️  Could not read: {image_path}")
        return
    
    result = detector.detect(img)
    
    print(f"\n  📸 {image_path.name}")
    print(f"     Detections : {result['detection_count']}")
    for d in result["detections"]:
        print(f"       - {d['class_name']:12s}  conf={d['confidence']:.2f}")
    print(f"     All Clear  : {result['all_clear']}")
    for alert in result["alerts"]:
        print(f"     {alert}")
    
    # Save annotated output
    out_path = save_dir / f"result_{image_path.name}"
    cv2.imwrite(str(out_path), result["annotated_frame"])
    print(f"     Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PPE Detection Test")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--conf",  type=float, default=0.35,  help="Confidence threshold")
    parser.add_argument("--limit", type=int,   default=10,    help="Max images to test")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PPE Detection - Model Test")
    print("=" * 60)
    
    detector = PPEDetector(conf=args.conf)
    
    save_dir = PROJECT_ROOT / "runs" / "test_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        test_single(detector, Path(args.image), save_dir)
    else:
        # Test on sample images from test set
        test_dir = PROJECT_ROOT / "dataset" / "images" / "test"
        images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + \
                 list(test_dir.glob("*.png"))
        
        if not images:
            print(f"No images found in {test_dir}")
            return
        
        print(f"Testing on {min(args.limit, len(images))} images from {test_dir}")
        print(f"Saving results to {save_dir}\n")
        
        stats = {"all_clear": 0, "alerts": 0, "total": 0}
        
        for img_path in images[:args.limit]:
            test_single(detector, img_path, save_dir)
            result = detector.detect(cv2.imread(str(img_path)))
            stats["total"] += 1
            if result["all_clear"]:
                stats["all_clear"] += 1
            else:
                stats["alerts"] += 1
        
        print("\n" + "=" * 60)
        print(f"  Tested: {stats['total']} images")
        print(f"  ✅ Compliant : {stats['all_clear']}")
        print(f"  ⚠️  Violations: {stats['alerts']}")
        print(f"  Results saved to: {save_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
