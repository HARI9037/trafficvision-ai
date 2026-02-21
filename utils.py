import cv2
import supervision as sv
import numpy as np
from dataclasses import dataclass

@dataclass
class VideoConfig:
    source: str  # Path to video file or '0' for webcam
    width: int
    height: int
    fps: int

def get_video_config(source_path: str) -> VideoConfig:
    """Retrieves video configuration from the source."""
    if source_path == "0":
        # Webcam
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30 # Default webcam FPS, can be unreliable
        cap.release()
        return VideoConfig(source=source_path, width=width, height=height, fps=fps)
    else:
        # File
        video_info = sv.VideoInfo.from_video_path(source_path)
        return VideoConfig(
            source=source_path,
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps
        )

def resize_frame(frame: np.ndarray, width: int = 1280) -> np.ndarray:
    """Resizes a frame to the specified width while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    
    aspect_ratio = width / w
    new_height = int(h * aspect_ratio)
    return cv2.resize(frame, (width, new_height))
