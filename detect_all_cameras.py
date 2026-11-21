"""
Detect all available cameras and test them
"""
import cv2

print("=" * 60)
print("CAMERA DETECTION - Testing all USB cameras")
print("=" * 60)

available_cameras = []

# Test camera indices 0-4
for i in range(5):
    print(f"\nTesting Camera {i}...", end=" ")
    
    # Try DirectShow backend (Windows)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"✓ FOUND - Resolution: {width}×{height}")
            available_cameras.append({
                'index': i,
                'width': width,
                'height': height,
                'backend': 'DirectShow'
            })
        else:
            print("✗ Cannot read frame")
        cap.release()
    else:
        print("✗ Not available")

print("\n" + "=" * 60)
print(f"SUMMARY: Found {len(available_cameras)} camera(s)")
print("=" * 60)

if available_cameras:
    for cam in available_cameras:
        print(f"Camera {cam['index']}: {cam['width']}×{cam['height']} ({cam['backend']})")
    
    print("\n" + "=" * 60)
    print("To use a specific camera with YOLO:")
    print("=" * 60)
    for cam in available_cameras:
        print(f"  Camera {cam['index']}: python yolo_simple.py  # Edit CAMERA_INDEX to {cam['index']}")
else:
    print("\n⚠ No cameras detected!")
    print("Make sure your USB camera is connected.")

print("\n" + "=" * 60)
