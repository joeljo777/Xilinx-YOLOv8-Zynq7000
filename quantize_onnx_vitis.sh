#!/bin/bash
# Vitis-AI ONNX Quantization for YOLOv8n
# Run this script INSIDE the Vitis-AI Docker container
# Uses ONNX path (more reliable for YOLOv8)

set -e  # Exit on error

echo "============================================================"
echo "  VITIS-AI ONNX QUANTIZATION FOR YOLOv8n"
echo "============================================================"

# Step 1: Activate Vitis-AI environment
echo ""
echo "[1/5] Activating Vitis-AI environment..."
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-optimizer-onnx
echo "✓ Environment activated: $(conda info --envs | grep '*')"

# Step 2: Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install opencv-python-headless numpy --quiet
echo "✓ Dependencies installed"

# Step 3: Verify files
echo ""
echo "[3/5] Verifying required files..."
if [ ! -f "yolov8n.onnx" ]; then
    echo "❌ Error: yolov8n.onnx not found!"
    exit 1
fi
if [ ! -d "calibration_data" ]; then
    echo "❌ Error: calibration_data/ directory not found!"
    exit 1
fi
echo "✓ Found yolov8n.onnx ($(du -h yolov8n.onnx | cut -f1))"
echo "✓ Found calibration_data/ ($(ls calibration_data/*.jpg 2>/dev/null | wc -l) images)"

# Step 4: Prepare calibration dataset
echo ""
echo "[4/5] Preparing calibration dataset..."
python3 << 'PREP_SCRIPT'
import cv2
import numpy as np
from pathlib import Path
import json

def prepare_calibration_data():
    """Prepare calibration images for Vitis-AI ONNX quantizer"""
    calib_dir = Path("calibration_data")
    output_dir = Path("calib_images_preprocessed")
    output_dir.mkdir(exist_ok=True)
    
    image_files = sorted(list(calib_dir.glob('*.jpg')))[:200]
    print(f"Processing {len(image_files)} calibration images...")
    
    calib_list = []
    
    for idx, img_file in enumerate(image_files):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(image_files)}")
        
        # Load and preprocess
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        
        # Save preprocessed image
        output_path = output_dir / f"calib_{idx:04d}.npy"
        img_normalized = img.astype(np.float32) / 255.0
        img_chw = img_normalized.transpose(2, 0, 1)  # HWC -> CHW
        np.save(output_path, img_chw)
        
        calib_list.append(str(output_path))
    
    # Save calibration list
    with open('calib_list.txt', 'w') as f:
        f.write('\n'.join(calib_list))
    
    print(f"\n✓ Prepared {len(calib_list)} calibration images")
    print(f"✓ Saved to: {output_dir}/")
    return len(calib_list)

num_images = prepare_calibration_data()
print(f"\n✓ Calibration dataset ready: {num_images} images")
PREP_SCRIPT

# Step 5: Run Vitis-AI ONNX quantization
echo ""
echo "[5/5] Running Vitis-AI quantization (10-15 minutes)..."
echo "This will quantize yolov8n.onnx to INT8 format for DPU..."

python3 << 'QUANT_SCRIPT'
import os
os.makedirs("quantized_output", exist_ok=True)

print("=" * 60)
print("NOTE: Vitis-AI ONNX quantization for YOLO models")
print("=" * 60)
print()
print("The Vitis-AI ONNX quantizer may have limitations with YOLO")
print("detection heads. If this fails, we'll use the vai_q_pytorch")
print("approach or compile the ONNX directly using the DPU compiler.")
print()
print("Attempting ONNX quantization...")
print("=" * 60)

# Try using vai_q_onnx if available
try:
    from vai_q_onnx import quantize_static
    from onnxruntime.quantization import CalibrationDataReader
    import numpy as np
    
    class CalibrationReader(CalibrationDataReader):
        def __init__(self, calib_list_file):
            with open(calib_list_file, 'r') as f:
                self.calib_files = [line.strip() for line in f.readlines()]
            self.enum_data = None
        
        def get_next(self):
            if self.enum_data is None:
                self.enum_data = iter(self.calib_files)
            
            try:
                calib_file = next(self.enum_data)
                data = np.load(calib_file)
                return {'images': data[np.newaxis, ...].astype(np.float32)}
            except StopIteration:
                return None
    
    reader = CalibrationReader('calib_list.txt')
    
    quantize_static(
        'yolov8n.onnx',
        'quantized_output/yolov8n_int8.onnx',
        reader,
        quant_format='QDQ',
        activation_type='QInt8',
        weight_type='QInt8'
    )
    
    print("\n✓ ONNX quantization successful!")
    print("Output: quantized_output/yolov8n_int8.onnx")
    
except ImportError as e:
    print(f"\n⚠ vai_q_onnx not available: {e}")
    print("\nFalling back to direct ONNX compilation for DPU...")
    print("Your ONNX model can be compiled directly with vai_c_onnx")
    print("without pre-quantization.")
    
except Exception as e:
    print(f"\n⚠ Quantization encountered an issue: {e}")
    print("\nAlternative: Compile ONNX directly for DPU")
    print("The DPU compiler can handle FP32 ONNX models and quantize")
    print("during compilation.")

QUANT_SCRIPT

# Summary
echo ""
echo "============================================================"
echo "  ✅ PROCESS COMPLETE!"
echo "============================================================"
echo ""
echo "Output directory contents:"
ls -lh quantized_output/ 2>/dev/null || echo "No quantized_output yet"
echo ""
ls -lh *.onnx
echo ""
echo "Next Steps:"
echo "  1. If quantization succeeded: Use quantized_output/*.onnx"
echo "  2. If not: Use original yolov8n.onnx with vai_c_onnx compiler"
echo ""
echo "To compile for DPU, you need:"
echo "  - DPU architecture fingerprint from your teammate"
echo "  - Run: vai_c_onnx --model yolov8n.onnx --arch DPUCZDX8G_ISA1_B4096"
echo ""
echo "Exit container with: exit"
echo "============================================================"
