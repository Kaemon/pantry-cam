# Pantry Cam

Real-time office pantry monitoring system using face recognition and object detection. Tracks who enters the pantry, how long they stay, and when the coffee machine is in use — all logged automatically and visualised in an interactive timeline.

## Demo

<!-- Add screenshots here — blur faces before uploading -->
<!-- ![Timeline](docs/timeline.png) -->
<!-- ![Live view](docs/live.png) -->

## Features

- **Face re-identification** — recognises enrolled people using InsightFace ArcFace embeddings with a voting-based confirmation system (resists brief misidentification from angle changes)
- **Multi-person tracking** — BoT-SORT tracks multiple people simultaneously; merges duplicate tracks when the same person is recognised under different IDs
- **Pantry zone logging** — logs entry/exit times and duration for each person
- **Coffee machine detection** — custom-trained YOLOv11 model detects when a cup is placed on the machine; logs usage duration
- **Daily analysis** — session merging, per-person total time, peak hours, interactive Gantt timeline

## Tech Stack

| Component | Library |
|---|---|
| Face detection & embedding | [InsightFace](https://github.com/deepinsight/insightface) (ArcFace) |
| Multi-object tracking | [BoT-SORT](https://github.com/NirAharon/BoT-SORT) via Ultralytics |
| Cup detection | YOLOv11 (custom trained) |
| Camera stream | OpenCV |
| Analysis & visualisation | Plotly, Matplotlib |

## Project Structure

```
pantry_cam.py          # Main entry point — camera loop, tracking, zone logic, logging
face_tracker.py        # Face re-ID: ArcFace embedding, voting, enrollment management
coffee_tracker.py      # Cup detection: YOLO inference, ROI, state machine, logging
analyze_logs.py        # Daily log analysis and interactive timeline chart
train_cup.py           # Train YOLOv11 cup detector from labeled dataset
capture_training.py    # Capture training images from camera (coffee ROI crop)
labelme_to_yolo.py     # Convert LabelMe annotations to YOLO format
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/pantry-cam.git
cd pantry-cam
python -m venv venv
source venv/bin/activate
pip install ultralytics insightface onnxruntime opencv-python plotly matplotlib
```

## Setup

### 1. Enroll faces

Run the main script and press **E** to enroll a person:
```bash
python pantry_cam.py
```
Point the camera at each person's face and type their name. The more photos from different angles the better. Embeddings are saved to `enrollment_previews/`.

### 2. Draw the pantry zone ROI

On first run you will be prompted to click 4 points defining the pantry zone boundary. Press **Enter** to confirm. Saved to `pantry_roi.json`.

### 3. Draw the coffee machine ROI

Press **C** while running to set up the coffee machine region. Click 4 points around the cup placement area. Saved to `coffee_roi.json`.

### 4. Train the cup detector (optional)

Capture training images:
```bash
python capture_training.py   # press Q when done
```
Label with [LabelMe](https://github.com/labelmeai/labelme), then convert and train:
```bash
python labelme_to_yolo.py
python train_cup.py
```
Update `CUP_MODEL_PATH` in `coffee_tracker.py` to point to the new weights.

## Running

```bash
python pantry_cam.py
```

**Keyboard shortcuts while running:**

| Key | Action |
|---|---|
| `E` | Enroll a new face |
| `R` | Rebuild enrollment from `enrollment_previews/` folder |
| `C` | Reconfigure coffee machine ROI |
| `Q` | Quit |

Logs are written to `pantry_log_YYYY-MM-DD.csv` and `coffee_log_YYYY-MM-DD.csv`.

## Analysis

```bash
python analyze_logs.py              # today
python analyze_logs.py 2026-06-15   # specific date
```

Output:
- Total time per person in pantry
- Peak hours (unique people present per hour)
- Interactive Gantt timeline (opens in browser) showing every session per person alongside coffee machine usage

## How It Works

### Face Re-ID
Each tracked person's bounding box crop is sent to a background thread. InsightFace extracts a 512-dim ArcFace embedding, which is compared against all enrolled embeddings using cosine similarity. A **voting window** (1 second, 51% majority) prevents brief angle changes from locking the wrong name. Changing a confirmed name requires 90% majority over 2 seconds.

### Session Merging
BoT-SORT can lose a track momentarily when someone is occluded (e.g. another person walks past). The analysis script merges consecutive visits by the same person with gaps under 2 minutes into a single session, giving accurate total-time statistics.

### Coffee Machine Detection
A YOLOv11 model trained on ~180 images of cups in the specific camera angle detects cup presence inside the ROI. A 3-second confirmation window and 5-second grace period prevent false logs from brief occlusions.
