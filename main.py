import cv2
import numpy as np
import torch
import csv
import os
from collections import deque


# Helper function to calculate distance between bounding boxes
def calculate_distance(box1, box2):
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


# Function to group persons that are close to each other
def group_persons(person_boxes, threshold=50):
    groups = []
    for i, box1 in enumerate(person_boxes):
        group = [box1]
        for j, box2 in enumerate(person_boxes):
            if i != j and calculate_distance(box1, box2) < threshold:
                group.append(box2)
        groups.append(group)
    return groups


# Track groups across frames
def track_groups(groups, tracked_groups, persistence_threshold=10):
    for group in groups:
        group_key = tuple(sorted([tuple(box) for box in group]))  # Using bounding box as the group key
        if group_key in tracked_groups:
            tracked_groups[group_key] += 1  # Increment frame count for group
        else:
            tracked_groups[group_key] = 1  # Start tracking this group

    crowds = [group for group, count in tracked_groups.items() if count >= persistence_threshold]
    return crowds, tracked_groups


# Log crowd detection events to a CSV file
def log_crowd_to_csv(crowds, frame_count, filename="results/crowd_detection_log.csv"):
    # Ensure the results folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for crowd in crowds:
            # Log the frame number and the number of persons in the crowd
            writer.writerow([frame_count, len(crowd)])


# Main function
def main():
    video_path = "data/people_video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    print("Processing video for crowd detection...")

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the pre-trained YOLOv5 small model

    tracked_groups = {}  # To track groups over frames
    frame_count = 0

    # Open the CSV file and write the header (only once)
    csv_file_path = "results/crowd_detection_log.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Person Count"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run YOLOv5 on the current frame
        results = model(frame)  # Perform detection on the frame
        person_boxes = []
        for *box, conf, cls in results.xyxy[0]:
            if cls == 0 and conf >= 0.5:  # Only consider persons (class 0)
                x1, y1, x2, y2 = map(int, box)  # Convert to integer values
                person_boxes.append((x1, y1, x2, y2))

        # Group persons that are close to each other
        groups = group_persons(person_boxes)

        # Track the groups across frames
        crowds, tracked_groups = track_groups(groups, tracked_groups)

        # Log detected crowds to the CSV file
        if crowds:
            log_crowd_to_csv(crowds, frame_count, filename=csv_file_path)

        # Display video with detected persons (optional)
        for group in groups:
            for box in group:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
