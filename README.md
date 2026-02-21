# Traffic Monitoring and Speed Estimation System

A real-time traffic monitoring system using YOLOv8 for vehicle detection, ByteTrack for tracking, and computer vision techniques for speed estimation.

## Features
- ✅ Real-time vehicle detection (Cars, Motorcycles, Buses, Trucks)
- ✅ Multi-object tracking with unique IDs
- ✅ Speed estimation (km/h)
- ✅ Vehicle counting and statistics
- ✅ Annotated video output
- ✅ **CSV export of tracked vehicles**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run with default video
```bash
python main.py
```

### Run with custom video
```bash
python main.py --source path/to/video.mp4
```

### Run with webcam
```bash
python main.py --source 0
```

## Output Files
1. **output.mp4** - Annotated video with bounding boxes, labels, speeds, and statistics
2. **tracked_vehicles.csv** - Detailed tracking data for each vehicle

## CSV Output Format
| Column | Description |
|--------|-------------|
| Vehicle_ID | Unique tracking ID |
| Vehicle_Type | Car, Motorcycle, Bus, or Truck |
| Max_Speed_kmh | Maximum speed recorded |
| Frame_Count | Number of frames detected |
| First_Seen_Frame | First appearance frame |
| Last_Seen_Frame | Last appearance frame |

## Finding Diverse Traffic Videos
For testing with buses, trucks, and motorcycles:
- [Pexels - Traffic Videos](https://www.pexels.com/search/videos/traffic/)
- [Coverr - Traffic Footage](https://coverr.co/s/traffic)
- [Vecteezy - Stock Videos](https://www.vecteezy.com/free-videos/traffic)

## Limitations
- Speed calibration is approximate (0.05 m/pixel)
- No perspective correction
- Webcam FPS may be unreliable

## Project Structure
```
├── main.py              # Main application
├── detector.py          # YOLOv8 vehicle detection
├── tracker.py           # ByteTrack wrapper
├── speed_estimator.py   # Speed calculation
├── visualizer.py        # Video annotation
├── utils.py             # Helper functions
└── requirements.txt     # Dependencies
```

## Controls
- Press `q` to stop video processing early

## Example Results
```
============================================================
TRACKING SUMMARY
============================================================
Total unique vehicles tracked: 2
Vehicle counts by type:
  - Car: 2
  - Motorcycle: 0
  - Bus: 0
  - Truck: 0

Tracking data exported to: tracked_vehicles.csv
Output video saved to: output.mp4
============================================================
```
👨‍💻 Contributors

This project was developed by:

Sreehari R Nair

Mirza Hamza Baig

Gautham S

NavaneethKrishna S
