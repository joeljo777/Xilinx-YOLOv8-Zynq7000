#!/bin/bash
# Simplified PyTorch Quantization for YOLOv8n
# Works with vitis-ai-pytorch environment

set -e

echo "============================================================"
echo "  VITIS-AI PYTORCH QUANTIZATION FOR YOLOv8n"
echo "============================================================"

# Activate pytorch environment
echo ""
echo "[1/6] Activating Vitis-AI PyTorch environment..."
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
echo "✓ Environment: vitis-ai-pytorch"

# Install dependencies
echo ""
echo "[2/6] Installing dependencies..."
pip install ultralytics opencv-python-headless --quiet
echo "✓ Dependencies installed"

# Verify files
echo ""
echo "[3/6] Verifying files..."
[ -f "yolov8n.pt" ] || { echo "❌ yolov8n.pt not found!"; exit 1; }
[ -d "calibration_data" ] || { echo "❌ calibration_data/ not found!"; exit 1; }
echo "✓ Found yolov8n.pt"
echo "✓ Found calibration_data/ ($(ls calibration_data/*.jpg 2>/dev/null | wc -l) images)"

# Check pytorch_nndct availability
echo ""
echo "[4/6] Checking quantization tools..."
python3 -c "from pytorch_nndct.apis import torch_quantizer; print('✓ pytorch_nndct available')" 2>&1

# Create simplified quantization script
echo ""
echo "[5/6] Preparing model for quantization..."

cat > simple_quantize.py << 'PYEOF'
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os

print("\nLoading YOLOv8n model...")
model = YOLO("yolov8n.pt")

# Get the PyTorch model
torch_model = model.model.float()
torch_model.eval()

print(f"Model loaded: {torch_model.__class__.__name__}")
print(f"Model parameters: {sum(p.numel() for p in torch_model.parameters())/1e6:.2f}M")

# Load calibration images
print("\nLoading calibration images...")
calib_dir = Path("calibration_data")
image_files = sorted(list(calib_dir.glob('*.jpg')))[:100]  # Use 100 for faster calibration

calibration_data = []
for idx, img_file in enumerate(image_files):
    if idx % 20 == 0:
        print(f"  Loading: {idx}/{len(image_files)}")
    
    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    calibration_data.append(img)

calibration_data = np.array(calibration_data)
print(f"\n✓ Loaded {len(calibration_data)} calibration images")

# Try quantization
try:
    from pytorch_nndct.apis import torch_quantizer
    
    print("\n" + "="*60)
    print("Running Vitis-AI quantization...")
    print("="*60)
    
    os.makedirs("quantize_result", exist_ok=True)
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Calibration mode
    print("\nStep 1: Calibration mode...")
    quantizer = torch_quantizer(
        quant_mode='calib',
        module=torch_model,
        input_args=(dummy_input,),
        output_dir='quantize_result',
        device=torch.device('cpu')
    )
    
    quant_model = quantizer.quant_model
    
    # Run calibration
    print("Running calibration with dataset...")
    with torch.no_grad():
        for i, img_batch in enumerate(calibration_data):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(calibration_data)}")
            img_tensor = torch.from_numpy(img_batch).unsqueeze(0)
            try:
                _ = quant_model(img_tensor)
            except Exception as e:
                print(f"  Warning at image {i}: {e}")
                continue
    
    quantizer.export_quant_config()
    print("\n✓ Calibration complete!")
    
    # Test mode
    print("\nStep 2: Generating quantized model...")
    quantizer = torch_quantizer(
        quant_mode='test',
        module=torch_model,
        input_args=(dummy_input,),
        output_dir='quantize_result',
        device=torch.device('cpu')
    )
    
    # IMPORTANT: Run forward pass on test model before exporting
    print("Running forward pass on quantized model...")
    quant_test_model = quantizer.quant_model
    with torch.no_grad():
        test_input = torch.from_numpy(calibration_data[0]).unsqueeze(0)
        _ = quant_test_model(test_input)
    print("✓ Test inference complete")
    
    # Export XIR/xmodel
    print("\nStep 3: Exporting to DPU format (.xmodel)...")
    quantizer.export_xmodel(deploy_check=False, output_dir='quantize_result')
    
    print("\n" + "="*60)
    print("✅ QUANTIZATION SUCCESSFUL!")
    print("="*60)
    print("\nOutput directory: quantize_result/")
    
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\nThe pytorch_nndct module is not available in this container.")
    print("This is needed for DPU quantization.")
    
except Exception as e:
    print(f"\n⚠ Quantization failed: {e}")
    print("\nThis might be due to YOLO model complexity.")
    print("Alternative: Your ONNX model can be deployed directly with reduced precision.")

print("\nDone!")
PYEOF

echo "✓ Quantization script created"

# Run quantization
echo ""
echo "[6/6] Running quantization..."
python3 simple_quantize.py

# Summary
echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
echo ""
if [ -d "quantize_result" ]; then
    echo "Output files:"
    ls -lh quantize_result/
    echo ""
    if ls quantize_result/*.xmodel 1> /dev/null 2>&1; then
        echo "✅ Found .xmodel file(s) - Ready for DPU deployment!"
    else
        echo "ℹ️  No .xmodel found. Check quantize_result/ for output files."
    fi
fi
echo ""
echo "Next: Exit container ('exit') and check quantize_result/"
echo "============================================================"
