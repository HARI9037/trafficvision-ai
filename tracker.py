import supervision as sv
from supervision import Detections

class VehicleTracker:
    def __init__(self):
        # Initialize ByteTrack
        self.tracker = sv.ByteTrack()

    def update(self, detections: Detections) -> Detections:
        """
        Updates the tracker with new detections.
        Returns detections with tracker_id assigned.
        """
        detections = self.tracker.update_with_detections(detections)
        return detections
