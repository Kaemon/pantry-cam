"""
Capture training images for the coffee cup detector.
Saves a cropped frame (same ROI + padding as inference) every second.

Usage:
    python capture_training.py
    Place / remove cup repeatedly. Press Q to stop.
"""
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CAMERA_URL   = os.getenv("RTSP_CAM1")
ROI_CONFIG   = "coffee_roi.json"
OUT_DIR      = Path("training_images")
PADDING      = 300   # must match CUP_CROP_PAD in coffee_tracker.py
INTERVAL_SEC = 1.0


def load_roi():
    if not Path(ROI_CONFIG).exists():
        raise FileNotFoundError(f"{ROI_CONFIG} not found — set up coffee ROI first (press C in pantry_cam.py)")
    data = json.load(open(ROI_CONFIG))
    return [tuple(p) for p in data["points"]]


def get_crop(frame, roi):
    fh, fw = frame.shape[:2]
    fsx, fsy = fw / 960, fh / 540
    xs = [p[0] for p in roi]
    ys = [p[1] for p in roi]
    x1 = max(0, int(min(xs) * fsx) - PADDING)
    y1 = max(0, int(min(ys) * fsy) - PADDING)
    x2 = min(fw, int(max(xs) * fsx) + PADDING)
    y2 = min(fh, int(max(ys) * fsy) + PADDING)
    return frame[y1:y2, x1:x2]


def main():
    roi = load_roi()
    OUT_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("ERROR: Cannot connect to camera")
        return

    print(f"Connected. Saving to {OUT_DIR}/  — Press Q to stop")
    saved      = 0
    last_save  = 0.0
    win        = "Capture Training (Q to quit)"

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        crop = get_crop(frame, roi)
        now = time.time()

        if now - last_save >= INTERVAL_SEC:
            ts = int(now * 1000)
            path = OUT_DIR / f"{ts}.jpg"
            cv2.imwrite(str(path), crop)
            saved += 1
            last_save = now
            print(f"\r  Saved: {saved}  ({path.name})", end="", flush=True)

        # Preview
        preview = cv2.resize(crop, (480, 360)) if crop.size > 0 else crop
        cv2.putText(preview, f"Saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {saved} images → {OUT_DIR}/")


if __name__ == "__main__":
    main()
