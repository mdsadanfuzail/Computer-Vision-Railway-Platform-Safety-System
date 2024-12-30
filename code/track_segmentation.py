from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

model_path_yolov11n_seg = base_dir / "weights" / "fine-tuned YOLOV11n-seg weights" / "best.pt"

if model_path_yolov11n_seg.exists():
    track_model = YOLO(model_path_yolov11n_seg)
else:
    print(f"The YOLO model file does not exist at {model_path_yolov11n_seg}")

def segment_tracks(frame):
    results = track_model(frame)
    if results is None or not hasattr(results[0], 'masks') or results[0].masks is None:
        print("No masks found in the model output.")
        return []
    track_masks = results[0].masks.data.cpu().numpy()
    contours_list = []

    for track_mask in track_masks:
        track_mask = (track_mask*255).astype(np.uint8)
        contours, _ = cv2.findContours(track_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

    return contours_list
