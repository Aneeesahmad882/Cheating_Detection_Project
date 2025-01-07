import cv2
from datetime import datetime
from facenet_pytorch import MTCNN
from ultralytics import YOLO
import os

# Initialize models
face_detector = MTCNN(keep_all=True)
object_detector = YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found or cannot be accessed.")
    exit()

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

def log_cheating(event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('logs/cheating_log.txt', 'a') as log_file:
        log_file.write(f"{timestamp} - {event}\n")

def save_snapshot(frame, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'logs/{label}_{timestamp}.jpg'
    cv2.imwrite(filename, frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    try:
        # Face detection
        faces, _ = face_detector.detect(frame)
        if faces is not None and len(faces) > 1:
            log_cheating("Multiple faces detected")
            save_snapshot(frame, "multiple_faces")
    except Exception as e:
        print(f"Error in face detection: {e}")

    try:
        # Object detection
        results = object_detector(frame)
        for result in results:
            if 'phone' in result.names:
                log_cheating("Mobile phone detected")
                save_snapshot(frame, "mobile_phone")
    except Exception as e:
        print(f"Error in object detection: {e}")

    # Display the frame
    cv2.imshow('Cheating Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
