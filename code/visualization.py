import numpy as np
import cv2

def draw_person_boxes(frame, person_boxes):
    for box in person_boxes:
        x1, y1, x2, y2 =map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, "person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def draw_tracks(frame, contours_list):
    track_mask_overplay = np.zeros_like(frame[:,:,0])
    for contour in contours_list:
        cv2.drawContours(track_mask_overplay, contour, -1, 255, thickness=cv2.FILLED)
    
    frame_with_mask = cv2.addWeighted(frame, 0.7, cv2.merge([track_mask_overplay]*3), 0.3, 0)
    return frame_with_mask

