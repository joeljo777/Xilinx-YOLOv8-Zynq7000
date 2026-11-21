"""
Quick YOLO test with Camera 0 (USB camera)
Press 'q' to quit
"""
from ultralytics import YOLO
import cv2

# Configuration
CAMERA_INDEX = 1  # <<< USB Camera (SeeCAM)
MODEL_PATH = "yolov8n.pt"

print("=" * 60)
print(f"YOLOv8n Real-Time Detection - USB Camera (Camera {CAMERA_INDEX})")
print("=" * 60)
print("Loading model...")

# Load model
model = YOLO(MODEL_PATH)
print(f"✓ Model loaded: {MODEL_PATH}")

# Open camera
print(f"Opening Camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"❌ Error: Cannot open Camera {CAMERA_INDEX}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✓ Camera opened: {width}×{height}")
print("\nPress 'q' to quit")
print("=" * 60)

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Run detection
        results = model.predict(source=frame, imgsz=640, verbose=False, device='cpu')
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Display
        cv2.imshow(f'YOLOv8n - Camera {CAMERA_INDEX}', annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} - Running...", end='\r')
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Processed {frame_count} frames")
    print("Done!")
