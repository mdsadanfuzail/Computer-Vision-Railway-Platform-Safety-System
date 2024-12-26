import numpy as np
import cv2

def check_proximity(person_boxes, contours_list, frame, safety_threshold = 50):
    danger_flags = []
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        person_center = ((x1 + x2)//2, (y1 + y2)//2)
        is_danger = False

        if contours_list: # If contour exists, then check proximity 
            for contour in contours_list:
                contour = contour[0]
                if contour.size == 0: # Skip empty contours
                    continue
                distance = cv2.pointPolygonTest(contour, person_center, True)
                if distance >= 0 and distance < safety_threshold:
                    is_danger = True
                    cv2.putText(frame, "DANGER!", person_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.circle(frame, person_center, 5, (0,0,255), -1)
                    break
            danger_flags.append(is_danger)

    return danger_flags    
