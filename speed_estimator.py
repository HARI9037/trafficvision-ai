import numpy as np
from collections import deque, defaultdict
import cv2

class SpeedEstimator:
    def __init__(self, fps: int, calibration_factor: float = 0.1):
        """
        calibration_factor: Meters per pixel.
        This is a simplified assumption. Ideally requires perspective transform.
        """
        self.fps = fps
        self.calibration_factor = calibration_factor
        # Store history of centroids: track_id -> deque of (x, y)
        self.history = defaultdict(lambda: deque(maxlen=fps)) 
        self.speeds = {} # track_id -> speed (km/h)

    def update(self, detections):
        """
        Updates speed estimates based on tracked detections.
        """
        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            if tracker_id is None:
                continue

            # Calculate centroid
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            self.history[tracker_id].append((cx, cy))
            
            if len(self.history[tracker_id]) > self.fps // 2: # Need at least 0.5s of data
                self.estimate_speed(tracker_id)
        
        return self.speeds

    def estimate_speed(self, track_id):
        # Calculate distance traveled in pixels
        # Simple Euclidean distance between first and last point in history
        # (For more accuracy, sum of segments or movement along a specific axis could be used)
        
        start_pos = self.history[track_id][0]
        end_pos = self.history[track_id][-1]
        
        pixel_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Convert to meters
        distance_meters = pixel_distance * self.calibration_factor
        
        # Time elapsed
        time_seconds = len(self.history[track_id]) / self.fps
        
        if time_seconds > 0:
            speed_mps = distance_meters / time_seconds
            speed_kmh = speed_mps * 3.6
            self.speeds[track_id] = int(speed_kmh)
        else:
            self.speeds[track_id] = 0
