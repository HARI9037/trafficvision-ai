import supervision as sv
import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def draw(self, frame: np.ndarray, detections: sv.Detections, speeds: dict) -> np.ndarray:
        labels = []
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            speed = speeds.get(tracker_id, 0)
            class_name = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}.get(class_id, "Unknown")
            labels.append(f"#{tracker_id} {class_name} {speed} km/h")

        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        
        # Add a stats panel (simple text for now)
        # In a real app, this could be more sophisticated
        
        return annotated_frame
