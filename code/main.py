import cv2
from pathlib import Path
from person_detection import detect_person
from track_segmentation import segment_tracks
from proximity_check import check_proximity
from visualization import draw_person_boxes, draw_tracks
import pygame

base_dir = Path(__file__).resolve().parent.parent

video_path = base_dir / "assets" / "test3.mp4"
output_path = base_dir/ "assets" / "saved" / "test3.mp4"

safety_threshold = 50

# for alarm sound
pygame.mixer.init()
alarm =  base_dir/ "assets" / "alarm" / "danger-alarm.mp3"
pygame.mixer.music.load(alarm)

#load video or camera
cap = cv2.VideoCapture(video_path)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect all people
    person_boxes, results = detect_person(frame)

    # Segment all the present tracks
    contours_list = segment_tracks(frame)

    # Check distance of people from tracks
    danger_flags = check_proximity(person_boxes, contours_list, frame, safety_threshold)
    if True in danger_flags:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play() # sound alarm

    # Visualizing the result
    draw_person_boxes(frame, person_boxes)
    frame_with_mask = draw_tracks(frame, contours_list)

    # Write frame to save
    out.write(frame_with_mask)

    # Display the frame
    cv2.imshow("Railway Platform Safety System", frame_with_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop.")
        break


# Release
cap.release()
out.release()
cv2.destroyAllWindows()