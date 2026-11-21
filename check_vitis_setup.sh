#!/bin/bash
# Direct DPU Compilation for YOLOv8n ONNX
# This compiles ONNX directly to .xmodel without separate quantization step

set -e

echo "============================================================"
echo "  VITIS-AI: COMPILE ONNX TO XMODEL FOR DPU"
echo "============================================================"

# Check available conda environments
echo ""
echo "Checking available Vitis-AI environments..."
conda info --envs

# Activate the pytorch environment (since we're in pytorch-cpu container)
echo ""
echo "[1/4] Activating Vitis-AI PyTorch environment..."
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
echo "✓ Activated: vitis-ai-pytorch"

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install opencv-python-headless numpy onnx onnxruntime --quiet
echo "✓ Dependencies installed"

# Verify files
echo ""
echo "[3/4] Verifying files..."
if [ ! -f "yolov8n.onnx" ]; then
    echo "❌ Error: yolov8n.onnx not found!"
    exit 1
fi
echo "✓ Found yolov8n.onnx ($(du -h yolov8n.onnx | cut -f1))"

# Check for vai_c_onnx compiler
echo ""
echo "[4/4] Checking for ONNX compiler..."
if command -v vai_c_onnx &> /dev/null; then
    echo "✓ vai_c_onnx compiler found"
    echo ""
    echo "To compile your ONNX model to .xmodel, you need the DPU architecture"
    echo "from your teammate. Ask them to run this on the Zynq board:"
    echo "  dexplorer -w"
    echo ""
    echo "Example DPU architectures:"
    echo "  - DPUCZDX8G_ISA1_B4096 (Zynq-7000 with 4096 RAM)"
    echo "  - DPUCZDX8G_ISA0_B512 (Zynq-7000 with 512 RAM)"
    echo ""
    echo "Once you have the architecture string, run:"
    echo "  vai_c_onnx --model yolov8n.onnx --arch DPUCZDX8G_ISA1_B4096 --output_dir compiled/"
else
    echo "⚠ vai_c_onnx not found in this container"
    echo ""
    echo "This PyTorch container doesn't include the ONNX compiler."
    echo "You have two options:"
    echo ""
    echo "Option 1: Use Vitis-AI tensorflow container (has vai_c_onnx)"
    echo "  docker pull xilinx/vitis-ai-tensorflow2-cpu:latest"
    echo ""
    echo "Option 2: Use Vitis-AI pytorch quantizer (recommended)"
    echo "  Let me create a PyTorch-based quantization script..."
fi

echo ""
echo "============================================================"
echo "  NEXT STEPS"
echo "============================================================"
echo ""
echo "1. Contact your teammate for DPU architecture string"
echo "2. Either:"
echo "   a) Use vai_c_onnx if available (TensorFlow container)"
echo "   b) Use PyTorch quantization (current container)"
echo ""
echo "I'll create a PyTorch quantization script for you now..."
echo "============================================================"
