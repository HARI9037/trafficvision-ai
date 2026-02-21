print("Starting diagnostic...")
try:
    import cv2
    print("cv2 imported")
    import supervision as sv
    print("supervision imported")
    from ultralytics import YOLO
    print("ultralytics imported")
    import numpy as np
    print("numpy imported")
    
    print("Attempting to load YOLO model...")
    model = YOLO("yolov8n.pt")
    print("YOLO model loaded")
    
    print("Diagnostic finished successfully")
except Exception as e:
    print(f"Error during diagnostic: {e}")
