# 🚀 2-DAY VITIS-AI DEPLOYMENT GUIDE

**Project**: Real-Time Object Detection - Laptop to FPGA  
**Timeline**: 2 Days  
**Current Status**: Webcam validation complete ✅

---

## ⚡ DAY 1: DATASET + BASELINE + EXPORT

### **Morning (2-3 hours)**

#### Step 1: Download COCO Dataset
```powershell
python download_coco.py
```
**Time**: 30-60 minutes (depending on internet speed)  
**Downloads**: ~1.2GB (5,000 validation images + annotations)

---

#### Step 2: Measure Baseline Accuracy
```powershell
python measure_accuracy.py
```
**Time**: 15-30 minutes (uses 1,000 sampled images for speed)  
**Output**: 
- Baseline mAP scores
- 200 calibration images saved to `calibration_data/`
- Results saved to `baseline_accuracy.txt`

**Expected mAP50**: ~0.450 (45%) for YOLOv8n on COCO

---

#### Step 3: Export to ONNX
```powershell
python export_to_onnx.py
```
**Time**: 2-3 minutes  
**Output**: `yolov8n.onnx` (ONNX model for Vitis-AI)

---

### **Afternoon (2-3 hours)**

#### Step 4: Install Docker Desktop

**Windows Installation**:
1. Download: https://www.docker.com/products/docker-desktop
2. Install Docker Desktop
3. Restart computer
4. Verify: `docker --version`

**Alternative - WSL2 Setup** (if Docker fails):
```powershell
# Enable WSL2
wsl --install

# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 20.04
wsl --install -d Ubuntu-20.04
```

---

#### Step 5: Pull Vitis-AI Docker Image
```powershell
docker pull xilinx/vitis-ai-pytorch-cpu:latest
```
**Time**: 30-60 minutes (image is ~5GB)

---

#### Step 6: Test Docker Container
```powershell
# Navigate to project directory
cd "c:\Users\josab\davinci media\Xilinx_project"

# Run Vitis-AI container
docker run -it --rm -v ${PWD}:/workspace xilinx/vitis-ai-pytorch-cpu:latest
```

**Inside container, verify**:
```bash
conda activate vitis-ai-pytorch
python --version  # Should show Python 3.7+
```

**Exit**: Type `exit`

---

## ⚡ DAY 2: QUANTIZATION + COMPILATION

### **Morning (2-3 hours)**

#### Step 7: Prepare Quantization Script

Create `quantize_model.py` (I'll provide this):
```python
# Quantizes ONNX model using calibration images
# Outputs INT8 model for DPU
```

---

#### Step 8: Run Quantization Inside Docker
```powershell
# Start Docker container
docker run -it --rm -v ${PWD}:/workspace xilinx/vitis-ai-pytorch-cpu:latest

# Inside container:
conda activate vitis-ai-pytorch
cd /workspace
python quantize_model.py
```

**Time**: 30-60 minutes  
**Output**: `yolov8n_int8.xmodel` (quantized model)

---

#### Step 9: Validate Quantized Model
```bash
# Inside Docker
python validate_quantized.py
```

**Check**: mAP should be >95% of baseline  
**Example**: If baseline was 0.450, quantized should be >0.427

---

### **Afternoon (2-3 hours)**

#### Step 10: Compile for Zynq DPU

**Select your DPU architecture**:
```bash
# List available architectures
ls /opt/vitis_ai/compiler/arch/

# Common options:
# - DPUCZDX8G_ISA1_B4096.json (Zynq-7000)
# - DPUCZDX8G_ISA0_B4096.json (Older Zynq)
```

**Compile**:
```bash
vai_c_xir \
  --xmodel yolov8n_int8.xmodel \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G_ISA1_B4096.json \
  --output_dir compiled/ \
  --net_name yolov8n
```

**Time**: 5-10 minutes  
**Output**: `compiled/yolov8n.xmodel` (DPU-ready model)

---

#### Step 11: Create Deployment Package

**Copy these files for Zynq deployment**:
```
deployment_package/
├── yolov8n.xmodel          # Compiled model
├── zynq_inference.py       # Inference script (I'll create)
├── coco_classes.txt        # Class names
└── README_deploy.md        # Deployment instructions
```

---

#### Step 12: Transfer to Zynq Board

**Via SCP** (if board is ready):
```bash
scp -r deployment_package/ root@<ZYNQ_IP>:/home/root/
```

**Via SD Card**:
- Copy `deployment_package/` to SD card
- Insert into Zynq board

---

## 📋 TIMELINE SUMMARY

### **Day 1** (6-7 hours total)
- ✅ 9:00 - 10:00: Download COCO
- ✅ 10:00 - 10:30: Measure accuracy
- ✅ 10:30 - 10:40: Export ONNX
- ☕ 10:40 - 11:00: Break
- ✅ 11:00 - 12:00: Install Docker
- 🍔 12:00 - 13:00: Lunch
- ✅ 13:00 - 14:00: Pull Vitis-AI image
- ✅ 14:00 - 15:00: Test setup

### **Day 2** (6-7 hours total)
- ✅ 9:00 - 10:00: Prepare quantization
- ✅ 10:00 - 11:00: Run quantization
- ✅ 11:00 - 11:30: Validate results
- ☕ 11:30 - 12:00: Break
- 🍔 12:00 - 13:00: Lunch
- ✅ 13:00 - 13:30: Compile for DPU
- ✅ 13:30 - 15:00: Create deployment package
- ✅ 15:00 - 16:00: Documentation + testing

---

## 🎯 SUCCESS CRITERIA

**End of Day 1**:
- ✅ COCO dataset downloaded
- ✅ Baseline mAP measured (~0.45)
- ✅ ONNX model exported
- ✅ Docker + Vitis-AI running

**End of Day 2**:
- ✅ Model quantized (INT8)
- ✅ Quantized mAP > 0.95 × baseline
- ✅ Compiled XMODEL ready
- ✅ Deployment package created

---

## 🚨 TROUBLESHOOTING

### Docker Issues
```powershell
# If Docker won't start:
# 1. Enable virtualization in BIOS
# 2. Enable Hyper-V in Windows Features
# 3. Restart PC

# Alternative: Use WSL2 instead
wsl --install
```

### Quantization Errors
```bash
# If calibration fails:
# - Reduce calibration images to 100
# - Ensure images are 640x640
# - Check ONNX model is valid
```

### Compilation Errors
```bash
# If DPU compilation fails:
# - Verify DPU architecture matches your board
# - Check Vitis-AI version compatibility
# - Try different compiler options
```

---

## 📞 CHECKPOINT QUESTIONS

**After each step, verify**:
- Did it complete without errors?
- Are output files created?
- Do file sizes look reasonable?

**If stuck**, stop and ask for help!

---

## 🎓 LEARNING RESOURCES

- **Vitis-AI Docs**: https://docs.xilinx.com/r/en-US/ug1414-vitis-ai
- **YOLO ONNX Export**: https://docs.ultralytics.com/modes/export/
- **Docker Basics**: https://docs.docker.com/get-started/

---

## ✅ READY TO START!

**Begin with**:
```powershell
python download_coco.py
```

**Track your progress in this order**:
1. ✅ Download COCO
2. ✅ Measure accuracy
3. ✅ Export ONNX
4. ✅ Install Docker
5. ✅ Pull Vitis-AI
6. ✅ Quantize model
7. ✅ Compile for DPU
8. ✅ Create deployment package

---

**Good luck! 🚀**
