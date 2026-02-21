import cv2
import argparse
import csv
from collections import defaultdict
from utils import get_video_config, resize_frame
from detector import VehicleDetector
from tracker import VehicleTracker
from speed_estimator import SpeedEstimator
from visualizer import Visualizer
import supervision as sv

# Vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASS_MAP = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

def main(source_path, headless=False):
    print(f"[DEBUG] Entering main with source_path={source_path}, headless={headless}")
    
    # Track detailed vehicle information
    print("[DEBUG] Initializing vehicle_details...")
    vehicle_details = defaultdict(lambda: {
        'vehicle_type': None,
        'max_speed': 0,
        'frame_count': 0,
        'first_seen': None,
        'last_seen': None
    })
    
    # Initialize components
    print("[DEBUG] Getting video config...")
    config = get_video_config(source_path)
    print(f"[DEBUG] Video config retrieved: {config}")
    
    print("[DEBUG] Initializing VehicleDetector...")
    detector = VehicleDetector()
    print("[DEBUG] VehicleDetector initialized.")
    
    print("[DEBUG] Initializing VehicleTracker...")
    tracker = VehicleTracker()
    print("[DEBUG] VehicleTracker initialized.")
    
    print("[DEBUG] Initializing SpeedEstimator...")
    speed_estimator = SpeedEstimator(fps=config.fps, calibration_factor=0.05)
    print("[DEBUG] SpeedEstimator initialized.")
    
    print("[DEBUG] Initializing Visualizer...")
    visualizer = Visualizer()
    print("[DEBUG] Visualizer initialized.")

    # Statistics
    print("[DEBUG] Initializing statistics...")
    vehicle_counts = {class_id: 0 for class_id in VEHICLE_CLASS_MAP.keys()}
    counted_ids = set()

    # Video Capture
    print(f"[DEBUG] Opening VideoCapture for {source_path}...")
    if source_path == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source_path}")
        return
    print("[DEBUG] VideoCapture opened successfully.")

    out = None
    frame_number = 0
    print("[DEBUG] Getting total frame count...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[DEBUG] Total frames: {total_frames}")

    print("[DEBUG] Starting frame processing loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] End of video reached or cannot read frame.")
            break
        
        frame_number += 1
        if frame_number % 10 == 0:
            print(f"Processing frame {frame_number}...")
            
        frame = resize_frame(frame, width=1280)
        
        # Initialize VideoWriter once
        if out is None and source_path != "0":
            print("[DEBUG] Initializing VideoWriter...")
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, config.fps, (w, h))
            print("[DEBUG] VideoWriter initialized.")

        # Detect
        detections = detector.detect(frame)
        
        # Track
        detections = tracker.update(detections)
        
        # Update Statistics and Vehicle Details
        if detections.tracker_id is not None:
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                if tracker_id is None:
                    continue
                if tracker_id not in counted_ids:
                    counted_ids.add(tracker_id)
                    if class_id in vehicle_counts:
                        vehicle_counts[class_id] += 1
                
                class_name = VEHICLE_CLASS_MAP.get(class_id, "Unknown")
                vehicle_details[tracker_id]['vehicle_type'] = class_name
                vehicle_details[tracker_id]['frame_count'] += 1
                if vehicle_details[tracker_id]['first_seen'] is None:
                    vehicle_details[tracker_id]['first_seen'] = frame_number
                vehicle_details[tracker_id]['last_seen'] = frame_number
        
        # Estimate Speed
        speeds = {}
        if detections.tracker_id is not None:
            speeds = speed_estimator.update(detections)
            for tracker_id, speed in speeds.items():
                if tracker_id is not None:
                    if speed > vehicle_details[tracker_id]['max_speed']:
                        vehicle_details[tracker_id]['max_speed'] = speed
        
        # Visualize
        if detections.tracker_id is not None:
            annotated_frame = visualizer.draw(frame, detections, speeds)
        else:
            annotated_frame = frame
            
        # Draw stats
        y_offset = 30
        for class_id, count in vehicle_counts.items():
            class_name = VEHICLE_CLASS_MAP.get(class_id, "Unknown")
            text = f"{class_name}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Write frame to output video
        if out is not None and out.isOpened():
            out.write(annotated_frame)
            
        if not headless:
            try:
                cv2.imshow("Traffic Monitoring", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[DEBUG] 'q' pressed, exiting loop.")
                    break
            except Exception as e:
                print(f"[DEBUG] cv2.imshow error: {e}")
                pass
    
    print("Processing finished. Releasing resources...")
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Export vehicle tracking data to CSV
    csv_filename = 'tracked_vehicles.csv'
    print(f"Exporting tracking data to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Vehicle_ID', 'Vehicle_Type', 'Max_Speed_kmh', 'Frame_Count', 'First_Seen_Frame', 'Last_Seen_Frame']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        valid_details = {k: v for k, v in vehicle_details.items() if k is not None}
        for vehicle_id, details in sorted(valid_details.items()):
            writer.writerow({
                'Vehicle_ID': vehicle_id,
                'Vehicle_Type': details['vehicle_type'],
                'Max_Speed_kmh': details['max_speed'],
                'Frame_Count': details['frame_count'],
                'First_Seen_Frame': details['first_seen'],
                'Last_Seen_Frame': details['last_seen']
            })
    
    # Print summary
    print("\n" + "="*60)
    print("TRACKING SUMMARY")
    print("="*60)
    print(f"Total unique vehicles tracked: {len(valid_details)}")
    print(f"Vehicle counts by type:")
    for class_id, count in vehicle_counts.items():
        class_name = VEHICLE_CLASS_MAP.get(class_id, "Unknown")
        print(f"  - {class_name}: {count}")
    print(f"\nTracking data exported to: {csv_filename}")
    print(f"Output video saved to: output.mp4")
    print("="*60) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Monitoring System")
    parser.add_argument("--source", type=str, default="traffic.mp4", help="Path to video file or '0' for webcam (default: traffic.mp4)")
    parser.add_argument("--headless", action="store_true", help="Run without display window")
    args = parser.parse_args()
    main(args.source, headless=args.headless)
