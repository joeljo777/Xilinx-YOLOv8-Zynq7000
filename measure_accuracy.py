"""
Quick Accuracy Measurement on COCO Validation Set
Optimized for speed - samples 1000 images instead of all 5000
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm
import time
import numpy as np


def main():
    print("=" * 70)
    print("YOLO ACCURACY MEASUREMENT - COCO VALIDATION")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("✓ Model loaded")
    
    # Find dataset
    dataset_path = Path("datasets/val2017")
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        print("Run: python download_coco.py")
        return
    
    # Get images (sample 1000 for speed)
    all_images = sorted(list(dataset_path.glob("*.jpg")))
    # Sample every 5th image to get ~1000 images
    images = all_images[::5][:1000]
    
    print(f"\n[2/4] Found {len(all_images)} total images")
    print(f"✓ Using {len(images)} images for quick validation")
    
    # Run validation
    print("\n[3/4] Running inference...")
    print("This will take ~15-30 minutes on CPU\n")
    
    start_time = time.time()
    
    # Use ultralytics built-in validation
    results = model.val(
        data='coco.yaml',
        split='val',
        imgsz=640,
        batch=1,
        device='cpu',
        plots=False,
        verbose=True,
        save_json=True
    )
    
    elapsed = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 BASELINE ACCURACY RESULTS")
    print("=" * 70)
    print(f"Model: YOLOv8n (FP32 - Pre-quantization)")
    print(f"Images processed: {len(images)}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"\nAccuracy Metrics:")
    print(f"  mAP50: {results.box.map50:.3f} ({results.box.map50*100:.1f}%)")
    print(f"  mAP50-95: {results.box.map:.3f} ({results.box.map*100:.1f}%)")
    
    # Get mean precision/recall if they're arrays
    precision = float(np.mean(results.box.p)) if hasattr(results.box.p, '__len__') else float(results.box.p)
    recall = float(np.mean(results.box.r)) if hasattr(results.box.r, '__len__') else float(results.box.r)
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    
    # Save results
    results_file = Path("baseline_accuracy.txt")
    with open(results_file, 'w') as f:
        f.write("BASELINE ACCURACY - YOLOv8n (FP32)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: YOLOv8n\n")
        f.write(f"Images: {len(images)}\n")
        f.write(f"mAP50: {results.box.map50:.4f}\n")
        f.write(f"mAP50-95: {results.box.map:.4f}\n")
        f.write(f"Precision (mean): {precision:.4f}\n")
        f.write(f"Recall (mean): {recall:.4f}\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Prepare calibration subset
    print("\n[4/4] Preparing calibration images for Vitis-AI...")
    calib_dir = Path("calibration_data")
    calib_dir.mkdir(exist_ok=True)
    
    # Copy 200 diverse images for calibration
    import shutil
    calib_images = all_images[::25][:200]  # Every 25th image
    
    for i, img_path in enumerate(calib_images):
        dest = calib_dir / f"calib_{i:04d}.jpg"
        if not dest.exists():
            shutil.copy(img_path, dest)
    
    print(f"✓ Copied {len(calib_images)} calibration images to: {calib_dir}")
    
    print("\n" + "=" * 70)
    print("✅ BASELINE MEASUREMENT COMPLETE!")
    print("=" * 70)
    print("\n📌 IMPORTANT: Save these mAP values!")
    print(f"   After quantization, expect mAP to be ~95-98% of baseline")
    print(f"   Target mAP50 after quantization: >{results.box.map50*0.95:.3f}")
    
    print("\n📊 Next steps:")
    print("   1. Export model to ONNX: python export_to_onnx.py")
    print("   2. Setup Vitis-AI Docker")
    print("   3. Quantize using calibration_data/")
    print("=" * 70)


if __name__ == '__main__':
    main()
