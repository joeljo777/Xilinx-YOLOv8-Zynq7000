import cv2

print("Checking available cameras...\n")

for i in range(5):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        print(f"Camera {i}: OK")
        # Try to get camera name
        backend = cap.getBackendName()
        print(f"  Backend: {backend}")
    else:
        print(f"Camera {i}: Not found")
    cap.release()

print("\nDone!")
