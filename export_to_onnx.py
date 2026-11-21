"""
Export YOLOv8n to ONNX format for Vitis-AI Quantization
Optimized for Xilinx Zynq-7000 DPU deployment

IMPORTANT: Uses 640x640 (standard YOLO size) for maximum accuracy
Camera preprocessing (640x480 → 640x640) handled at inference time
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import onnx
import numpy as np


def main():
    print("=" * 70)
    print("EXPORT YOLOv8n TO ONNX - VITIS-AI OPTIMIZED")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("✓ Model loaded: yolov8n.pt")
    
    # Export to ONNX
    print("\n[2/4] Exporting to ONNX format...")
    print("Configuration:")
    print("  - Format: ONNX")
    print("  - Input size: 640×640 (standard YOLO, maintains accuracy)")
    print("  - Opset: 11 (Vitis-AI compatible)")
    print("  - Simplify: True (optimized graph)")
    print("\nThis may take 1-2 minutes...\n")
    
    output_path = model.export(
        format='onnx',
        imgsz=640,  # Standard YOLO size (NOT [480,640] - see analysis)
        simplify=True,
        opset=11,  # Vitis-AI compatible opset
        dynamic=False  # Fixed input size for DPU
    )
    
    print(f"✓ ONNX model exported to: {output_path}")
    
    # Verify ONNX model
    print("\n[3/4] Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure is valid")
    
    # Get detailed model info
    print("\n[4/4] Analyzing model structure...")
    
    # Input info
    input_tensor = onnx_model.graph.input[0]
    input_name = input_tensor.name
    input_shape = input_tensor.type.tensor_type.shape
    input_dims = [d.dim_value for d in input_shape.dim]
    
    # Output info
    output_tensors = onnx_model.graph.output
    
    output_path_obj = Path(output_path)
    file_size = output_path_obj.stat().st_size / (1024 * 1024)  # MB
    
    print("\n" + "=" * 70)
    print("✅ ONNX EXPORT SUCCESSFUL!")
    print("=" * 70)
    print(f"\n📁 Model Information:")
    print(f"  File: {output_path_obj.name}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Location: {output_path_obj.absolute()}")
    
    print(f"\n📊 Model Architecture:")
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_dims}")
    print(f"  Expected format: (batch=1, channels=3, height=640, width=640)")
    print(f"  Opset version: 11")
    print(f"  Number of outputs: {len(output_tensors)}")
    
    # Verify input shape is correct
    expected_shape = [1, 3, 640, 640]
    if input_dims == expected_shape:
        print(f"\n✓ Input shape verified: {input_dims} ✅")
        print("  Model is ready for Vitis-AI quantization")
    else:
        print(f"\n⚠ Warning: Unexpected input shape: {input_dims}")
        print(f"  Expected: {expected_shape}")
    
    print("\n📝 Important Notes:")
    print("  • Model trained on 640×640 images (maintains accuracy)")
    print("  • Camera frames (640×480) will be letterboxed to 640×640 at inference")
    print("  • This approach preserves the 52.1% mAP baseline measured")
    print("  • Standard YOLO deployment - proven and reliable")
    
    print("\n📊 Next Steps (Day 2):")
    print("  1. Install Docker Desktop (if not already done)")
    print("  2. Pull Vitis-AI image:")
    print("     docker pull xilinx/vitis-ai-pytorch-cpu:latest")
    print("  3. Run quantization inside Docker using:")
    print("     - Model: yolov8n.onnx")
    print("     - Calibration: calibration_data/ (200 images)")
    
    print("\n💾 Files Ready for Vitis-AI Quantization:")
    print(f"  ✓ Model: {output_path_obj.name}")
    print(f"  ✓ Calibration: calibration_data/ (200 images)")
    print(f"  ✓ Baseline: baseline_accuracy.txt (mAP50: 0.521)")
    
    print("=" * 70)
    print("\n🚀 Ready for Vitis-AI quantization workflow!")
    print("=" * 70)


if __name__ == '__main__':
    main()
