"""
Vitis-AI Quantization Script for YOLOv8n
Converts ONNX model to INT8 quantized .xmodel for Xilinx DPU
"""

import torch
import numpy as np
from pytorch_nndct.apis import torch_quantizer
import cv2
import os
from pathlib import Path

def load_calibration_images(calib_dir, num_images=200, img_size=640):
    """
    Load calibration images for quantization
    """
    calib_path = Path(calib_dir)
    image_files = list(calib_path.glob('*.jpg'))[:num_images]
    
    print(f"Loading {len(image_files)} calibration images...")
    
    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        images.append(img)
    
    # Stack into batches
    images = np.array(images)
    print(f"Calibration data shape: {images.shape}")
    return images

def calibration_data_generator(calib_images, batch_size=1):
    """
    Generator function for calibration data
    """
    num_images = len(calib_images)
    for i in range(0, num_images, batch_size):
        batch = calib_images[i:i + batch_size]
        yield torch.from_numpy(batch)

# Placeholder YOLOv8n model structure (simplified for quantization)
class YOLOv8Wrapper(torch.nn.Module):
    """
    Wrapper for ONNX model to work with PyTorch quantizer
    """
    def __init__(self, onnx_path):
        super(YOLOv8Wrapper, self).__init__()
        # Load ONNX model using torch.onnx
        import onnx
        self.onnx_model = onnx.load(onnx_path)
        print(f"Loaded ONNX model from {onnx_path}")
        
    def forward(self, x):
        # This is a placeholder - actual forward pass handled by ONNX runtime
        # For Vitis-AI quantization, we need PyTorch model
        # You may need to convert ONNX back to PyTorch or use native PyTorch model
        return x

def main():
    print("=" * 60)
    print("VITIS-AI QUANTIZATION FOR YOLOv8n")
    print("=" * 60)
    
    # Configuration
    ONNX_MODEL = "yolov8n.onnx"
    CALIB_DIR = "calibration_data"
    OUTPUT_DIR = "quantized_model"
    BATCH_SIZE = 1
    IMG_SIZE = 640
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load calibration data
    print("\n[1/4] Loading calibration images...")
    calib_images = load_calibration_images(CALIB_DIR, num_images=200, img_size=IMG_SIZE)
    
    # Load model
    print("\n[2/4] Loading model...")
    # For Vitis-AI, we should use the original PyTorch model
    # ONNX quantization requires different approach
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    pytorch_model = model.model  # Extract PyTorch model
    pytorch_model.eval()
    
    print(f"Model loaded: {type(pytorch_model)}")
    
    # Initialize quantizer
    print("\n[3/4] Initializing Vitis-AI quantizer...")
    quantizer = torch_quantizer(
        quant_mode='calib',  # 'calib' for calibration, 'test' for quantized inference
        module=pytorch_model,
        input_args=(torch.randn(1, 3, IMG_SIZE, IMG_SIZE),),
        output_dir=OUTPUT_DIR,
        device=torch.device('cpu')
    )
    
    # Run calibration
    print("\n[4/4] Running quantization calibration...")
    quant_model = quantizer.quant_model
    
    with torch.no_grad():
        for idx, batch in enumerate(calibration_data_generator(calib_images, BATCH_SIZE)):
            print(f"  Processing batch {idx + 1}/{len(calib_images)}", end='\r')
            _ = quant_model(batch)
    
    print("\n\nCalibration complete!")
    
    # Export quantized model
    print("\nExporting quantized model...")
    quantizer.export_quant_config()
    
    # Generate .xmodel (DPU-ready format)
    print("\nGenerating .xmodel for DPU deployment...")
    quantizer = torch_quantizer(
        quant_mode='test',
        module=pytorch_model,
        input_args=(torch.randn(1, 3, IMG_SIZE, IMG_SIZE),),
        output_dir=OUTPUT_DIR,
        device=torch.device('cpu')
    )
    
    # Deploy to DPU format
    quantizer.export_xmodel(deploy_check=False)
    
    print("\n" + "=" * 60)
    print("✅ QUANTIZATION COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}/")
    print("Look for .xmodel file - this is what you'll deploy to Zynq!")
    print("=" * 60)

if __name__ == "__main__":
    main()
