"""
Auto-detect and use the best available camera for YOLO inference
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time
import argparse
from ultralytics import YOLO


def find_working_camera():
    """Find the first working camera."""
    print("\n[INFO] Searching for working cameras...")
    
    for i in range(5):
        for backend_id, backend_name in [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF")]:
            cap = cv2.VideoCapture(i, backend_id)
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  ✓ Camera {i} working ({backend_name}): {width}x{height}")
                cap.release()
                return i, backend_id, backend_name
            cap.release()
    
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Real-Time YOLO Object Detection')
    parser.add_argument('--source', type=str, default='auto', help='Camera source (0, 1, 2, or "auto" to detect)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--display-fps', action='store_true', help='Display FPS')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("REAL-TIME OBJECT DETECTION - YOLO WEBCAM")
    print("=" * 60)
    
    # Load model
    print(f"\n[INFO] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print(f"[INFO] Model loaded successfully!")
    
    # Find camera
    if args.source == 'auto':
        cam_idx, backend, backend_name = find_working_camera()
        if cam_idx is None:
            print("[ERROR] No working cameras found!")
            return
    else:
        cam_idx = int(args.source)
        backend = cv2.CAP_DSHOW  # Try DirectShow first
        backend_name = "DirectShow"
    
    print(f"\n[INFO] Opening Camera {cam_idx} with {backend_name}...")
    cap = cv2.VideoCapture(cam_idx, backend)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera {cam_idx}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {width}x{height}")
    
    print("\n" + "=" * 60)
    print("Press 'q' to quit | Press 's' to save frame")
    print("=" * 60 + "\n")
    
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[WARNING] Failed to grab frame")
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # Run YOLO inference
            results = model(frame, conf=args.conf, device='cpu', verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            if time.time() - fps_start_time > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Display info
            num_detections = len(results[0].boxes)
            cv2.putText(annotated_frame, f'Detections: {num_detections}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if args.display_fps:
                cv2.putText(annotated_frame, f'FPS: {fps:.1f}', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('YOLO Object Detection - SeeCAM', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                filename = f'detection_frame_{frame_count}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"[INFO] Saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Processed {frame_count} frames")
        print("[INFO] Done!")


if __name__ == '__main__':
    main()
