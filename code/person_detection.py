import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)

def detect_person(frame):
    results = model(frame)

    detections = results.pandas().xyxy[0]

    # Filter for person class (class 0)
    person_boxes = detections[detections['class'] == 0][['xmin', 'ymin', 'xmax', 'ymax']].values

    return person_boxes, results