from ultralytics import YOLO
from pathlib import Path

base_dir = Path(__file__).resolve().parent
model_path_yolov8n = base_dir / "weights" / "pre-trained YOLOv8n weights" / "yolov8n.pt"

if not model_path_yolov8n.exists():
    print(f"The YOLO model file does not exist at {model_path_yolov8n}")
else:
    person_model = YOLO(model_path_yolov8n)

def detect_person(frame):
    results = person_model(frame, classes = [0])
    person_boxes = results[0].boxes.xyxy.cpu().numpy()
    return person_boxes, results