"""
Find SeeCAM Camera - Try all possible camera sources
"""

import cv2

print("=" * 70)
print("SEARCHING FOR SeeCAM_CU_CHL_TC_BX")
print("=" * 70)

found_cameras = []

# Try different backends
backends = [
    (cv2.CAP_MSMF, "MSMF (Microsoft Media Foundation)"),
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_ANY, "Auto"),
]

print("\nScanning all camera indices with different backends...\n")

for backend_id, backend_name in backends:
    print(f"[{backend_name}]")
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i, backend_id)
            ret, frame = cap.read()
            
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                camera_info = {
                    'index': i,
                    'backend': backend_name,
                    'backend_id': backend_id,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                }
                
                # Check if this is a new camera (not already found)
                if not any(c['index'] == i for c in found_cameras):
                    found_cameras.append(camera_info)
                    print(f"  ✓ Camera {i}: {width}x{height} @ {fps:.1f}fps")
                
            cap.release()
        except Exception as e:
            pass
    print()

print("=" * 70)
print(f"FOUND {len(found_cameras)} CAMERA(S)")
print("=" * 70)

if len(found_cameras) == 0:
    print("\n❌ No cameras detected!")
    print("\nTroubleshooting:")
    print("1. Make sure the SeeCAM is connected via USB")
    print("2. Check Windows Device Manager for the camera")
    print("3. Try unplugging and reconnecting the camera")
    print("4. Make sure no other application is using the camera")

elif len(found_cameras) == 1:
    print("\n⚠ Only 1 camera found (likely your laptop camera)")
    print(f"   Camera 0: {found_cameras[0]['resolution']}")
    print("\nSeeCAM not detected. Please:")
    print("1. Verify the SeeCAM USB connection")
    print("2. Check if Windows recognizes the device")

else:
    print(f"\n✓ Found {len(found_cameras)} cameras:")
    for cam in found_cameras:
        print(f"\n  Camera {cam['index']}:")
        print(f"    Resolution: {cam['resolution']}")
        print(f"    FPS: {cam['fps']:.1f}")
        print(f"    Backend: {cam['backend']}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    
    if len(found_cameras) >= 2:
        external_cam = found_cameras[1]  # Assume second camera is external
        print(f"\nCamera 0 = Laptop webcam")
        print(f"Camera {external_cam['index']} = Likely your SeeCAM (external USB)")
        print(f"\nTo test the SeeCAM, run:")
        print(f"  python test_camera_view.py")
        print(f"\nTo run YOLO detection on SeeCAM:")
        print(f"  python yolo_webcam_inference.py --source {external_cam['index']} --display-fps")
    
print("\n" + "=" * 70)
