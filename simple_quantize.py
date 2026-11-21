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
