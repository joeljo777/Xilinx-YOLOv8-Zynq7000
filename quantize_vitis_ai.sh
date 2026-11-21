#!/bin/bash
# Vitis-AI Quantization Workflow for YOLOv8n
# Run this script INSIDE the Vitis-AI Docker container

set -e  # Exit on error

echo "============================================================"
echo "  VITIS-AI QUANTIZATION FOR YOLOv8n"
echo "============================================================"

# Step 1: Activate Vitis-AI PyTorch environment
echo ""
echo "[1/6] Activating Vitis-AI PyTorch environment..."
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
echo "✓ Environment activated"

# Step 2: Install dependencies
echo ""
echo "[2/6] Installing dependencies..."
pip install ultralytics opencv-python-headless --quiet
echo "✓ Dependencies installed"

# Step 3: Verify files
echo ""
echo "[3/6] Verifying required files..."
if [ ! -f "yolov8n.pt" ]; then
    echo "❌ Error: yolov8n.pt not found!"
    exit 1
fi
if [ ! -d "calibration_data" ]; then
    echo "❌ Error: calibration_data/ directory not found!"
    exit 1
fi
echo "✓ Found yolov8n.pt"
echo "✓ Found calibration_data/ ($(ls calibration_data/*.jpg 2>/dev/null | wc -l) images)"

# Step 4: Export PyTorch model to TorchScript
echo ""
echo "[4/6] Exporting model to TorchScript..."
python3 << 'EXPORT_SCRIPT'
from ultralytics import YOLO
import torch

# Load YOLOv8n model
model = YOLO("yolov8n.pt")
pytorch_model = model.model
pytorch_model.eval()

# Export to TorchScript
dummy_input = torch.randn(1, 3, 640, 640)
traced_model = torch.jit.trace(pytorch_model, dummy_input)
traced_model.save("yolov8n_traced.pt")

print("✓ Model exported to yolov8n_traced.pt")
EXPORT_SCRIPT

# Step 5: Run Vitis-AI quantization
echo ""
echo "[5/6] Running Vitis-AI quantization (this takes 10-15 minutes)..."
python3 << 'QUANT_SCRIPT'
import torch
from pytorch_nndct.apis import torch_quantizer
import cv2
import numpy as np
from pathlib import Path
import os

# Load traced model
model = torch.jit.load("yolov8n_traced.pt")
model.eval()

# Prepare calibration data
def load_calibration_images(calib_dir="calibration_data", num_images=100):
    """Load and preprocess calibration images"""
    calib_path = Path(calib_dir)
    image_files = sorted(list(calib_path.glob('*.jpg')))[:num_images]
    
    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        images.append(img)
    
    return np.array(images)

print("Loading calibration images...")
calib_images = load_calibration_images(num_images=100)
print(f"Loaded {len(calib_images)} calibration images")

# Create quantizer - Calibration mode
print("\nInitializing quantizer (calibration mode)...")
os.makedirs("quantized_output", exist_ok=True)

quantizer = torch_quantizer(
    quant_mode='calib',
    module=model,
    input_args=(torch.randn(1, 3, 640, 640),),
    output_dir='quantized_output',
    device=torch.device('cpu')
)

# Run calibration
print("Running calibration...")
quant_model = quantizer.quant_model

with torch.no_grad():
    for i, img_batch in enumerate(calib_images):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(calib_images)} images")
        img_tensor = torch.from_numpy(img_batch).unsqueeze(0)
        _ = quant_model(img_tensor)

print("\nExporting calibration config...")
quantizer.export_quant_config()

# Test mode - Generate quantized model
print("\nGenerating quantized model (test mode)...")
quantizer = torch_quantizer(
    quant_mode='test',
    module=model,
    input_args=(torch.randn(1, 3, 640, 640),),
    output_dir='quantized_output',
    device=torch.device('cpu')
)

# Export to XIR format (.xmodel for DPU)
print("\nCompiling for DPU...")
quantizer.export_xmodel(deploy_check=False, output_dir='quantized_output')

print("\n✓ Quantization complete!")
print("Output files in: quantized_output/")
QUANT_SCRIPT

# Step 6: Summary
echo ""
echo "============================================================"
echo "  ✅ QUANTIZATION COMPLETE!"
echo "============================================================"
echo ""
echo "Output files:"
ls -lh quantized_output/*.xmodel 2>/dev/null || echo "Checking for .xmodel files..."
ls -lh quantized_output/ | grep -E '\.(xmodel|pt)$' || ls -lh quantized_output/
echo ""
echo "Next steps:"
echo "  1. Exit this container: type 'exit'"
echo "  2. Copy .xmodel to SD card"
echo "  3. Deploy to Zynq board with teammates"
echo ""
echo "============================================================"
