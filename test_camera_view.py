"""
Simple Camera Test - View camera feed in a window
Press 'q' to quit
"""

import cv2

# Try different camera sources
camera_source = 1  # Camera 1 for SeeCAM (external USB camera)

print("=" * 60)
print("CAMERA TEST VIEWER - SeeCAM_CU_CHL_TC_BX")
print("=" * 60)
print(f"Opening camera source: {camera_source}")
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to take a screenshot")
print("=" * 60)

# Open camera
cap = cv2.VideoCapture(camera_source, cv2.CAP_MSMF)

if not cap.isOpened():
    print(f"\n[ERROR] Could not open camera {camera_source}")
    print("Try changing camera_source to 1, 2, or 3 in the script")
    exit(1)

# Get camera info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"\n✓ Camera opened successfully!")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print("\nStarting video feed...\n")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[WARNING] Failed to grab frame")
            break
        
        frame_count += 1
        
        # Add info overlay
        cv2.putText(frame, f'Camera {camera_source}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Camera Test - SeeCAM Preview', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f'camera_test_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal frames captured: {frame_count}")
    print("Camera test complete!")
