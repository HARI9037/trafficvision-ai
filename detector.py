from ultralytics import YOLO
import supervision as sv
import numpy as np

class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence
        # COCO classes for vehicles: car(2), motorcycle(3), bus(5), truck(7)
        self.vehicle_classes = [2, 3, 5, 7]

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detects vehicles in the frame.
        Returns supervision.Detections object.
        """
        results = self.model(frame, verbose=False, conf=self.confidence)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for specific vehicle classes
        detections = detections[np.isin(detections.class_id, self.vehicle_classes)]
        
        return detections
