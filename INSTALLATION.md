# Installation Guide - Real-Time Object Detection

## System Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.8 or higher (3.9 recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional (NVIDIA GPU with CUDA support for faster inference)

---

## Step 1: Install Python Dependencies

### Option A: Using pip (Recommended)
```powershell
# Ensure pip is up to date
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# If you have an NVIDIA GPU with CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python numpy pandas ultralytics
```

### Option B: Using requirements.txt
```powershell
# Navigate to the project directory
cd "c:\Users\josab\davinci media\Xilinx_project"

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 2: Download YOLOv7-Tiny Weights

### Automatic Download (Recommended)
The script will automatically download the weights on first run. Just execute:
```powershell
python yolo_webcam_inference.py
```

### Manual Download
If you prefer to download manually:

**YOLOv7-Tiny (Recommended for Zynq):**
```powershell
# Download from official YOLOv7 repository
curl -L -o yolov7-tiny.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```

**Alternative - YOLOv8-Nano (Also Vitis-AI compatible):**
```powershell
# This will be downloaded automatically by ultralytics
# Or manually:
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## Step 3: Verify Webcam Access

### Test webcam (Optional)
```powershell
# Quick OpenCV webcam test
python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print('Webcam OK' if ret else 'Webcam FAILED'); cap.release()"
```

---

## Step 4: Run the Inference Script

### Basic Usage (Webcam 0)
```powershell
python yolo_webcam_inference.py
```

### With Custom Parameters
```powershell
# Use different webcam
python yolo_webcam_inference.py --source 1

# Custom confidence threshold (default: 0.25)
python yolo_webcam_inference.py --conf 0.5

# Display FPS counter
python yolo_webcam_inference.py --display-fps

# Use custom weights
python yolo_webcam_inference.py --weights yolov7-tiny.pt
```

### Full Command Example
```powershell
python yolo_webcam_inference.py --source 0 --weights yolov7-tiny.pt --conf 0.3 --iou 0.45 --img-size 640 --display-fps
```

---

## Controls During Execution
- **Press 'q'**: Quit the application
- **Press 's'**: Save the current frame as an image

---

## Troubleshooting

### Error: "Failed to open webcam"
- Check if another application is using the webcam
- Try a different source number: `--source 1` or `--source 2`
- On Windows, ensure camera permissions are enabled

### Error: "No module named 'torch'"
- Reinstall PyTorch: `pip install torch torchvision`

### Error: "CUDA out of memory"
- Reduce image size: `--img-size 416`
- Use CPU instead: Uninstall GPU PyTorch, install CPU version

### Slow Performance
- Use GPU if available (install CUDA version of PyTorch)
- Reduce image size: `--img-size 416` or `--img-size 320`
- Close other applications

---

## Model Compatibility Matrix

| Model | Size (MB) | Speed (FPS)* | mAP | Vitis-AI Ready |
|-------|-----------|--------------|-----|----------------|
| YOLOv7-Tiny | 12 | 60-120 | 38.7 | ✅ Yes |
| YOLOv8n | 6 | 80-150 | 37.3 | ✅ Yes |
| YOLOv5s | 14 | 50-100 | 37.4 | ✅ Yes |

*FPS varies by hardware

---

## Next Steps for Vitis-AI Deployment

1. ✅ **Current Stage**: Validate model on laptop (you are here)
2. 🔄 **Next**: Collect validation images for quantization
3. 🔄 **Next**: Quantize model using Vitis-AI PyTorch flow
4. 🔄 **Next**: Compile for Zynq-7000 DPU
5. 🔄 **Next**: Deploy to FPGA

---

## Quick Reference

### Directory Structure (After Running)
```
Xilinx_project/
├── yolo_webcam_inference.py    # Main inference script
├── requirements.txt             # Python dependencies
├── INSTALLATION.md              # This file
├── README.md                    # Dataset recommendations
├── yolov7-tiny.pt               # Model weights (auto-downloaded)
└── detection_frame_*.jpg        # Saved frames (if 's' pressed)
```

### Important File Paths
- **Script**: `yolo_webcam_inference.py`
- **Weights**: `yolov7-tiny.pt` (or `yolov8n.pt`)
- **Requirements**: `requirements.txt`

---

**Author**: Embedded AI Engineer  
**Date**: February 2026  
**Target Platform**: Xilinx Zynq-7000 with Vitis-AI
