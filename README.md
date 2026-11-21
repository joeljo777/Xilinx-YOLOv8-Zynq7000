# Real-Time Object Detection - Laptop to FPGA

**Project Type**: Embedded AI with Xilinx Vitis-AI  
**Target Hardware**: Zynq-7000 SoC (FPGA + ARM Cortex-A9)  
**Framework**: PyTorch → Vitis-AI Quantization → DPU Deployment

---

## Project Overview

This project implements a **real-time object detection pipeline** that:
1. ✅ Validates a Tiny-YOLO model on a laptop using webcam inference
2. 🔄 Quantizes the model for FPGA deployment using Vitis-AI
3. 🔄 Deploys to Xilinx Zynq-7000 DPU for edge inference

**Current Stage**: Laptop Validation (Webcam Inference)

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install torch torchvision opencv-python numpy pandas ultralytics
```

### 2. Run Webcam Inference
```powershell
python yolo_webcam_inference.py --display-fps
```

That's it! The model will auto-download and start detecting objects from your webcam.

---

## 📦 Recommended Datasets for Testing

Before deploying to the Zynq board, you should validate the model's accuracy on a standard dataset.

### **Option 1: COCO Validation Set (Recommended)**

**Why COCO?**
- YOLOv7-Tiny and YOLOv8n are pre-trained on COCO (80 classes)
- Industry-standard benchmark
- Perfect for evaluating accuracy before quantization

**Download Instructions:**
```powershell
# Create a datasets folder
mkdir datasets
cd datasets

# Download COCO 2017 validation images (1GB)
# Option A: Using curl (if available)
curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
Expand-Archive -Path val2017.zip -DestinationPath .

# Option B: Using wget (if available)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download COCO annotations
curl -L -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
Expand-Archive -Path annotations_trainval2017.zip -DestinationPath .
```

**Manual Download:**
- Images: https://cocodataset.org/#download
- Download "2017 Val images [1GB]"
- Download "2017 Train/Val annotations [241MB]"

**Dataset Stats:**
- **Images**: 5,000 validation images
- **Classes**: 80 object categories
- **Size**: ~1GB
- **Format**: JPEG images with JSON annotations

**File Structure:**
```
datasets/
├── val2017/
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ... (5,000 images)
└── annotations/
    └── instances_val2017.json
```

---

### **Option 2: Pascal VOC 2012 (Lightweight Alternative)**

**Why VOC?**
- Smaller dataset (500MB vs 1GB)
- Faster to download and process
- Good for initial testing

**Download Instructions:**
```powershell
mkdir datasets
cd datasets

# Download VOC 2012
curl -L -o VOCtrainval_11-May-2012.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Extract
tar -xf VOCtrainval_11-May-2012.tar
```

**Manual Download:**
- Link: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- Download "Training/Validation data"

**Dataset Stats:**
- **Images**: 11,540 images (5,717 val images)
- **Classes**: 20 object categories
- **Size**: ~2GB
- **Format**: JPEG + XML annotations

**File Structure:**
```
datasets/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
```

---

### **Option 3: Custom Validation Set (Minimal)**

If you just need quick testing without downloading large datasets:

**Create a small test set from your webcam:**
```powershell
# Run the script and press 's' to save frames
python yolo_webcam_inference.py
# Press 's' 20-30 times to save frames with various objects
```

Then organize them:
```powershell
mkdir datasets\custom_test
move detection_frame_*.jpg datasets\custom_test\
```

---

## 📊 Dataset Comparison Table

| Dataset | Images (Val) | Classes | Size | Download Time* | Best For |
|---------|--------------|---------|------|----------------|----------|
| **COCO 2017** | 5,000 | 80 | 1GB | 5-10 min | Production validation |
| **Pascal VOC** | 5,717 | 20 | 2GB | 10-15 min | Quick testing |
| **Custom** | 20-50 | 80 | <50MB | 0 min | Rapid prototyping |

*Assuming 10 Mbps connection

---

## 🔧 Modifying the Script for Image Folder Input

To test on a dataset folder instead of webcam, modify the `VideoSource` class or use this approach:

### Future Feature (Already Structured):
```python
# In the script, change:
video_source = VideoSource(source)

# To process a folder (future enhancement):
# image_folder = Path('datasets/val2017')
# for img_path in image_folder.glob('*.jpg'):
#     frame = cv2.imread(str(img_path))
#     detections = detector.detect(frame)
#     # ... process
```

### Quick Batch Processing Script:
```python
# Save this as batch_inference.py
import cv2
from pathlib import Path
from yolo_webcam_inference import YOLODetector

detector = YOLODetector('yolov7-tiny.pt')
image_folder = Path('datasets/val2017')

for img_path in list(image_folder.glob('*.jpg'))[:100]:  # Process 100 images
    frame = cv2.imread(str(img_path))
    detections = detector.detect(frame)
    print(f"{img_path.name}: {len(detections)} objects detected")
```

---

## 🎯 Recommended Dataset for Your Use Case

**For Vitis-AI Quantization Workflow:**

I recommend **COCO 2017 Validation Set** because:
1. ✅ Pre-trained model is already optimized for COCO
2. ✅ Accurate baseline metrics for comparison post-quantization
3. ✅ Vitis-AI calibration requires ~100-1000 images (COCO has 5,000)
4. ✅ Standard benchmark for reporting results
5. ✅ 80 classes cover most real-world scenarios

**Download Command (Easiest):**
```powershell
# Install helper tool
pip install fiftyone

# Download COCO validation set automatically
python -c "import fiftyone.zoo as foz; dataset = foz.load_zoo_dataset('coco-2017', split='validation')"
```

---

## 📁 Project Structure

```
Xilinx_project/
├── yolo_webcam_inference.py    # ← Main inference script (READY TO RUN)
├── requirements.txt             # ← Python dependencies
├── INSTALLATION.md              # ← Setup instructions
├── README.md                    # ← This file
│
├── datasets/                    # ← Place datasets here
│   ├── val2017/                 # COCO validation images
│   └── annotations/             # COCO annotations
│
├── yolov7-tiny.pt               # ← Auto-downloaded on first run
│
└── outputs/                     # ← Future: quantized models
    └── quantized_model.xmodel   # For Zynq deployment
```

---

## 🔄 Workflow Roadmap

### Phase 1: Laptop Validation ✅ (Current)
- [x] Install PyTorch and dependencies
- [x] Run webcam inference
- [x] Verify real-time detection
- [ ] Download COCO validation set
- [ ] Measure baseline accuracy (mAP)

### Phase 2: Vitis-AI Quantization 🔄 (Next)
- [ ] Install Vitis-AI Docker/Environment
- [ ] Quantize model (INT8)
- [ ] Validate quantized accuracy
- [ ] Export to XMODEL format

### Phase 3: Zynq Deployment 🔄 (Future)
- [ ] Compile XMODEL for DPU
- [ ] Flash Zynq board
- [ ] Deploy inference application
- [ ] Benchmark FPS on hardware

---

## 🛠️ Key Script Features

The `yolo_webcam_inference.py` script is designed with **Vitis-AI compatibility** in mind:

1. **PyTorch-based** (required for Vitis-AI quantizer)
2. **Modular design** (easy to swap webcam → image folder)
3. **YOLOv7-Tiny** (optimized for Zynq DPU)
4. **COCO class names** (80 classes)
5. **Real-time visualization** (bounding boxes + labels)

---

## 📚 Additional Resources

### Vitis-AI Documentation
- Vitis-AI GitHub: https://github.com/Xilinx/Vitis-AI
- Quantization Guide: https://docs.xilinx.com/r/en-US/ug1414-vitis-ai

### YOLO Models
- YOLOv7: https://github.com/WongKinYiu/yolov7
- YOLOv8: https://github.com/ultralytics/ultralytics

### Datasets
- COCO: https://cocodataset.org/
- Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/

---

## 🤝 Support

**Role Distribution:**
- **You (ML Lead)**: Model validation, accuracy benchmarking
- **Embedded AI Engineer**: Vitis-AI quantization, FPGA deployment

**Next Steps:**
1. Run the webcam script to verify it works
2. Download COCO validation set
3. Measure baseline mAP on 1,000 images
4. Share results before moving to quantization

---

