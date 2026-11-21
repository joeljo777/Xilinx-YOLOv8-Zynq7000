import cv2
import subprocess

print("=" * 60)
print("DETAILED CAMERA DETECTION")
print("=" * 60)

# Method 1: OpenCV detection with detailed info
print("\n[Method 1] OpenCV Camera Detection:")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)  # Use MSMF backend for Windows
    ret, frame = cap.read()
    if ret:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        backend = cap.getBackendName()
        
        print(f"\n✓ Camera {i} - DETECTED")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Backend: {backend}")
        
        # Try different backends to get more info
        cap.release()
        cap_dshow = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_dshow.isOpened():
            print(f"  DirectShow: Available")
        cap_dshow.release()
    else:
        # Still release even if not opened
        cap.release()

# Method 2: Windows Device Manager listing
print("\n" + "=" * 60)
print("[Method 2] Windows Device Manager (Imaging Devices):")
print("=" * 60)
try:
    result = subprocess.run(
        ['powershell', '-Command', 
         "Get-PnpDevice -Class 'Camera','Image' | Where-Object {$_.Status -eq 'OK'} | Select-Object FriendlyName, InstanceId | Format-Table -AutoSize"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
except Exception as e:
    print(f"Could not query device manager: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("If you see 'SeeCAM_CU_CHL_TC_BX' in the list above,")
print("look for its corresponding Camera number (0, 1, 2, etc.)")
print("\nThen run:")
print("  python yolo_webcam_inference.py --source <NUMBER> --display-fps")
print("=" * 60)
