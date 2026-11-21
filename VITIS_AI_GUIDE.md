# Vitis-AI Quantization Guide for YOLOv8n

## ✅ What You've Completed
- [x] Docker Desktop installed and running
- [x] ONNX model exported (yolov8n.onnx - 12.3MB)
- [x] Calibration data prepared (200 images in calibration_data/)
- [x] Baseline accuracy measured (mAP50: 0.521)

## 📋 Next Steps

### Step 1: Pull Vitis-AI Docker Image (⏱️ 30-60 min)
```powershell
docker pull xilinx/vitis-ai-pytorch-cpu:latest
```
**Note:** This downloads ~5GB and takes 30-60 minutes depending on your internet speed.

### Step 2: Run Vitis-AI Container
```powershell
# Navigate to your project directory
cd "c:\Users\josab\davinci media\Xilinx_project"

# Run Vitis-AI Docker container with your project mounted
docker run -it --rm `
  -v ${PWD}:/workspace `
  xilinx/vitis-ai-pytorch-cpu:latest `
  bash
```

**Inside the container, you'll be in a Linux environment with:**
- Ubuntu 20.04
- Python 3.7/3.8
- PyTorch 1.x
- Vitis-AI tools pre-installed

### Step 3: Install Dependencies (Inside Container)
```bash
# You're now inside the Docker container
cd /workspace

# Activate Vitis-AI PyTorch environment
conda activate vitis-ai-pytorch

# Install ultralytics
pip install ultralytics opencv-python
```

### Step 4: Run Quantization
```bash
# Still inside the container
python quantize_model.py
```

**Expected output:**
- Calibration process: ~10-15 minutes
- Output: `quantized_model/*.xmodel` file
- This .xmodel file is ready for Zynq DPU deployment!

### Step 5: Copy .xmodel to Host
```bash
# Check the output
ls -lh quantized_model/

# Your .xmodel file should be here
# Exit container (files persist because of volume mount)
exit
```

### Step 6: Transfer to SD Card
After exiting Docker:
```powershell
# Check your quantized model
ls quantized_model/

# Copy .xmodel to SD card (replace D: with your SD card drive)
Copy-Item "quantized_model\*.xmodel" -Destination "D:\"
```

## 🎯 Deployment with Teammates

### Files Your Teammates Already Have:
1. **BOOT.bin** - Bootloader with DPU configured for CNN acceleration
2. **image.ub** - Linux kernel image
3. **PetaLinux** - Root filesystem

### What You're Providing:
- **yolov8n_quantized.xmodel** - Your quantized YOLO model

### SD Card Structure Should Look Like:
```
SD Card (FAT32 partition):
├── BOOT.bin          (from teammate 1)
├── image.ub          (from teammate 2)
└── yolov8n.xmodel    (from you - quantized model)

SD Card (ext4 partition):
└── rootfs/           (PetaLinux from teammate 2)
```

## 🔧 Important Information Needed from Teammates

Before compilation, ask your teammates for:
1. **DPU Architecture String** (e.g., `DPUCZDX8G_ISA1_B4096`)
   - They can get this by running on the Zynq board:
   ```bash
   dexplorer -w
   ```

2. **DPU Configuration Details:**
   - Number of DPU cores
   - RAM size allocated to DPU
   - Clock frequency

## 📊 Expected Results

### Quantization Impact:
- **Original model (FP32):** mAP50 = 0.521 (52.1%)
- **Quantized model (INT8):** mAP50 ≈ 0.50-0.52 (minimal accuracy loss)
- **Model size:** 12.3MB → ~3-4MB
- **Inference speed on DPU:** Expected 15-30 FPS at 640x640

### Quality Check:
After quantization, verify:
```python
# Compare accuracy (optional)
# If mAP50 drops below 0.49, recalibrate with more images
```

## 🚨 Troubleshooting

### Issue: "Docker daemon is not running"
**Solution:** Open Docker Desktop from Start menu, wait for it to start

### Issue: "Cannot connect to Docker daemon"
**Solution:** Restart Docker Desktop or restart your PC

### Issue: "Out of disk space"
**Solution:** Vitis-AI needs ~10GB free. Clear space and try again

### Issue: "Quantization fails"
**Solution:** 
1. Ensure calibration_data/ has 200 images
2. Check yolov8n.pt exists in project folder
3. Verify ONNX model loads correctly

### Issue: "No .xmodel generated"
**Solution:** Check quantized_model/ directory for error logs

## 📝 Timeline Estimate

| Task | Duration |
|------|----------|
| Pull Docker image | 30-60 min |
| Run container & setup | 5 min |
| Quantization | 10-15 min |
| Testing | 5 min |
| **Total** | **~1 hour** |

## ✨ Final Checklist Before Deployment

- [ ] .xmodel file generated (check file size: should be 3-4MB)
- [ ] Docker container exited successfully
- [ ] .xmodel copied to SD card
- [ ] SD card has BOOT.bin + image.ub + .xmodel
- [ ] Teammates ready to test on Zynq board
- [ ] DPU architecture string confirmed

---

**Good luck! You're one hour away from FPGA deployment! 🚀**
