from typing import List

import cv2
import numpy as np
import pygame
from ultralytics import YOLO

from src.models.bounding_box import BoundingBox
from src.models.user import User
from src.sounds import audio
from src.storage import create_table_if_not_exists, insert_data

project_id = "sigma-drive-2"
dataset_id = "sigma_data"
table_id = "detections"

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
print("The model has been loaded")

# Open the video file
video_path = "videos/test_video.mp4"
cap = cv2.VideoCapture(video_path)

pygame.mixer.init()

targets = ["car", "person"]


def get_objects_coordinates(frame) -> List[BoundingBox]:
    predictions = model(frame)
    boxes = []
    for i, box in enumerate(predictions[0].boxes):
        # The method xyxy returns x_top_left, y_top_left, x_bottom_right, y_bottom_right
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = box.xyxy.tolist()[0]
        name_id = box.cls
        name = model.names[int(name_id)]
        box = BoundingBox(name, int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right))
        boxes.append(box)
    return boxes


def filter_objects(objects: List[BoundingBox], objects_names: List[str]) -> List[BoundingBox]:
    filtered_objects = []
    for object in objects:
        if object.name in objects_names:
            filtered_objects.append(object)
    return filtered_objects


def plot_objects(frame: np.ndarray, objects: List[BoundingBox]) -> np.ndarray:
    # Blue in BGR
    color = (255, 0, 0)
    thickness = 2
    for object in objects:
        # Draw the rectangle on the image
        frame = cv2.rectangle(frame, (object.x_top_left, object.y_top_left),
                              (object.x_bottom_right, object.y_bottom_right),
                              color, thickness)
    return frame


def play_sound(objects: List[BoundingBox]):
    # Conditional logic
    # Example in the case there's a stop sign
    audio_path = audio["stop"]
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()


user = User("user123", "Jean-Loïc", 33, "France")
table = create_table_if_not_exists(project_id, dataset_id, table_id)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        all_boxes = get_objects_coordinates(frame)
        filtered_boxes = filter_objects(all_boxes, targets)
        image = plot_objects(frame, filtered_boxes)
        cv2.imshow("Analyse", image)

        insert_data(user, filtered_boxes, table)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
# Generator of Results objects
cv2.destroyAllWindows()
