"""
Real-Time Object Detection - Laptop Webcam Inference
YOLOv7-Tiny on PyTorch (Vitis-AI Compatible)

Author: Embedded AI Engineer
Purpose: Validate model on laptop before Zynq deployment
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time
import argparse


class YOLODetector:
    """
    Wrapper class for YOLOv7-Tiny detection.
    Designed for easy input source switching (webcam -> image folder).
    """
    
    def __init__(self, weights_path='yolov7-tiny.pt', conf_threshold=0.25, iou_threshold=0.45, img_size=640):
        """
        Initialize the YOLO detector.
        
        Args:
            weights_path (str): Path to the .pt weights file
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            img_size (int): Input image size (640 recommended for YOLOv7-Tiny)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        # Force CPU for compatibility (torchvision NMS may not support CUDA in some builds)
        self.device = torch.device('cpu')
        
        print(f"[INFO] Using device: {self.device}")
        print(f"[INFO] Loading model from: {weights_path}")
        
        # Load the YOLOv7 model
        self.model = self._load_model(weights_path)
        
        # COCO class names (80 classes)
        self.class_names = self._get_coco_names()
        
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Detected {len(self.class_names)} classes")
        
    def _load_model(self, weights_path):
        """Load YOLOv7 model using torch.hub or local weights."""
        try:
            # Try loading from local file first
            if Path(weights_path).exists():
                model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights_path, trust_repo=True)
            else:
                # Download pretrained YOLOv7-Tiny from torch hub
                print(f"[INFO] Weights not found locally. Downloading YOLOv7-Tiny...")
                model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-tiny.pt', trust_repo=True)
                
            model.to(self.device)
            model.conf = self.conf_threshold
            model.iou = self.iou_threshold
            model.eval()
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print(f"[INFO] Falling back to ultralytics YOLOv8n (also Vitis-AI compatible)...")
            # Fallback to ultralytics YOLO
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # Nano version, very fast
            # Force CPU device for ultralytics
            model.to(self.device)
            return model
    
    def _get_coco_names(self):
        """Return COCO dataset class names (80 classes)."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect(self, frame):
        """
        Run inference on a single frame.
        
        Args:
            frame (numpy.ndarray): Input frame (BGR format from OpenCV)
            
        Returns:
            list: Detection results [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        # Note: ultralytics uses 'imgsz', YOLOv7 uses 'size'
        # Force CPU device for ultralytics to avoid CUDA NMS issues
        results = self.model(img_rgb, imgsz=self.img_size, verbose=False, device='cpu')
        
        # Parse results
        detections = []
        
        # Handle both YOLOv7 and ultralytics YOLO format
        if hasattr(results, 'pandas'):
            # YOLOv7 format
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                detections.append([
                    int(row['xmin']),
                    int(row['ymin']),
                    int(row['xmax']),
                    int(row['ymax']),
                    float(row['confidence']),
                    int(row['class']),
                    row['name']
                ])
        else:
            # Ultralytics YOLO format
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    detections.append([
                        int(x1), int(y1), int(x2), int(y2),
                        float(conf), cls_id,
                        self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                    ])
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): Detection results from detect()
            
        Returns:
            numpy.ndarray: Frame with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2, conf, cls_id, cls_name = det
            
            # Generate color based on class
            color = self._get_color(cls_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f'{cls_name}: {conf:.2f}'
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _get_color(self, class_id):
        """Generate consistent color for each class."""
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


class VideoSource:
    """
    Abstract video source handler.
    Easy to extend for webcam, video file, or image folder.
    """
    
    def __init__(self, source=0):
        """
        Initialize video source.
        
        Args:
            source (int or str): 0 for webcam, or path to video file/image folder
        """
        self.source = source
        self.is_webcam = isinstance(source, int) or source.isdigit()
        
        if self.is_webcam:
            self.cap = cv2.VideoCapture(int(source))
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open webcam {source}")
            print(f"[INFO] Webcam {source} opened successfully")
        else:
            # For future: handle video files or image folders
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video source: {source}")
            print(f"[INFO] Video source opened: {source}")
    
    def read(self):
        """Read next frame."""
        return self.cap.read()
    
    def release(self):
        """Release the video source."""
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Real-Time YOLO Object Detection')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--weights', type=str, default='yolov7-tiny.pt', help='Path to YOLO weights (.pt file)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--display-fps', action='store_true', help='Display FPS on screen')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Initialize detector
    detector = YOLODetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size
    )
    
    # Initialize video source
    video_source = VideoSource(source)
    
    print("\n" + "="*60)
    print("REAL-TIME OBJECT DETECTION - YOLO WEBCAM INFERENCE")
    print("="*60)
    print("[INFO] Press 'q' to quit")
    print("[INFO] Press 's' to save current frame")
    print("="*60 + "\n")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = video_source.read()
            
            if not ret:
                print("[WARNING] Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run detection
            detections = detector.detect(frame)
            
            # Draw detections
            frame = detector.draw_detections(frame, detections)
            
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Display info on frame
            info_text = f'Detections: {len(detections)}'
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if args.display_fps:
                fps_text = f'FPS: {fps:.1f}'
                cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLO Object Detection - Laptop Validation', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                save_path = f'detection_frame_{frame_count}.jpg'
                cv2.imwrite(save_path, frame)
                print(f"[INFO] Frame saved to: {save_path}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        video_source.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Processed {frame_count} frames")
        print("[INFO] Done!")


if __name__ == '__main__':
    main()
