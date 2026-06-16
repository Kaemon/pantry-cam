import cv2
import csv
import json
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

COFFEE_ROI_CONFIG = "coffee_roi.json"
COFFEE_LOG_FILE   = f"coffee_log_{datetime.now().strftime('%Y-%m-%d')}.csv"
CUP_MODEL_PATH    = "runs/detect/runs/detect/cup_model/cpu_v1/weights/best.pt"

CUP_CONF        = 0.30
CUP_CROP_PAD    = 300  # must match PADDING in capture_training.py
CUP_CONFIRM_SEC = 3.0
CUP_GRACE_SEC   = 5.0
DETECT_INTERVAL = 0.2

_log_lock = threading.Lock()


def init_coffee_log():
    if not Path(COFFEE_LOG_FILE).exists():
        with open(COFFEE_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["camera", "start_time", "end_time", "duration_sec"]
            )


class CoffeeTracker:
    """Detects cup inside coffee ROI using a trained YOLO model."""

    def __init__(self, camera_name: str):
        self.camera_name             = camera_name
        self.roi: list[tuple[int, int]] = []
        self.roi_confirmed           = False
        self.cup_start:          datetime | None = None
        self.cup_gone_since:     float   | None = None
        self._cup_confirm_since: float   | None = None
        self._last_detect_time:  float          = 0.0
        self._last_boxes: list[tuple[int,int,int,int]] = []
        self._person_last_seen:  float          = 0.0
        self._fsx:               float          = 1.0
        self._fsy:               float          = 1.0
        self._fx1:               int            = 0
        self._fy1:               int            = 0

        if Path(CUP_MODEL_PATH).exists():
            self._model = YOLO(CUP_MODEL_PATH)
            print(f"[Coffee] Cup model loaded: {CUP_MODEL_PATH}")
        else:
            self._model = None
            print(f"[Coffee] WARNING: model not found at {CUP_MODEL_PATH}")

    # ── ROI ──────────────────────────────────────────────────────────────────

    def load_roi(self) -> bool:
        if Path(COFFEE_ROI_CONFIG).exists():
            data = json.load(open(COFFEE_ROI_CONFIG))
            self.roi[:] = [tuple(p) for p in data["points"]]
            self.roi_confirmed = True
            print(f"Loaded Coffee ROI: {self.roi}")
            return True
        return False

    def save_roi(self):
        with open(COFFEE_ROI_CONFIG, "w") as f:
            json.dump({"points": self.roi}, f)
        print(f"Coffee ROI saved → {COFFEE_ROI_CONFIG}")

    def setup_roi(self, cam):
        self.roi.clear()
        self.roi_confirmed = False
        win = "Coffee Machine ROI Setup"

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.roi) < 4:
                self.roi.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.roi:
                self.roi.pop()

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        print("\n=== Coffee Machine ROI Setup ===")
        print("Left-click to add points (up to 4), right-click to undo.")
        print("Press ENTER to confirm | R to reset\n")

        while True:
            frame = cam.get_frame()
            if frame is None:
                cv2.waitKey(100)
                continue
            display = cv2.resize(frame, (960, 540))
            self.draw_overlay(display)
            hints = [
                f"Points: {len(self.roi)}/4  — left-click to add, right-click to undo",
                "Press ENTER to confirm  |  R to reset",
            ]
            for i, txt in enumerate(hints):
                cv2.putText(display, txt, (10, 510 - (len(hints) - 1 - i) * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(self.roi) >= 3:
                self.roi_confirmed = True
                self.save_roi()
                cv2.destroyWindow(win)
                return
            elif key == ord('r'):
                self.roi.clear()

    # ── Crop helper ───────────────────────────────────────────────────────────

    def _get_crop(self, frame: np.ndarray):
        if not self.roi_confirmed or len(self.roi) < 3:
            return None
        xs = [p[0] for p in self.roi]
        ys = [p[1] for p in self.roi]
        fh, fw = frame.shape[:2]
        self._fsx = fw / 960
        self._fsy = fh / 540
        self._fx1 = max(0, int(min(xs) * self._fsx))
        self._fy1 = max(0, int(min(ys) * self._fsy))
        fx2 = min(fw, int(max(xs) * self._fsx))
        fy2 = min(fh, int(max(ys) * self._fsy))
        crop = frame[self._fy1:fy2, self._fx1:fx2]
        return crop if crop.size > 0 else None

    # ── Person suppression ────────────────────────────────────────────────────

    def _person_in_roi(self, person_boxes: list) -> bool:
        if not person_boxes:
            return False
        xs = [p[0] for p in self.roi]
        ys = [p[1] for p in self.roi]
        margin = int(20 * self._fsx)
        rx1 = max(0, int(min(xs) * self._fsx) - margin)
        ry1 = max(0, int(min(ys) * self._fsy) - margin)
        rx2 = int(max(xs) * self._fsx) + margin
        ry2 = int(max(ys) * self._fsy) + margin
        for (px1, py1, px2, py2) in person_boxes:
            if px2 > rx1 and px1 < rx2 and py2 > ry1 and py1 < ry2:
                return True
        return False

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray,
               person_boxes: list | None = None) -> list[tuple[int,int,int,int]]:
        if self._model is None or not self.roi_confirmed:
            return self._last_boxes

        now_ts = time.time()
        if now_ts - self._last_detect_time < DETECT_INTERVAL:
            return self._last_boxes
        self._last_detect_time = now_ts

        # compute scale factors for coord conversion
        fh, fw = frame.shape[:2]
        self._fsx = fw / 960
        self._fsy = fh / 540
        self._get_crop(frame)  # still call to keep _fx1/_fy1 up to date

        if person_boxes and self._person_in_roi(person_boxes):
            self._person_last_seen = now_ts
        if now_ts - self._person_last_seen < 1.5:
            self._cup_confirm_since = None
            return self._last_boxes

        # crop with padding around ROI — matches how training images were captured
        fh, fw = frame.shape[:2]
        xs = [p[0] for p in self.roi]
        ys = [p[1] for p in self.roi]
        pad = CUP_CROP_PAD
        icx1 = max(0, int(min(xs) * self._fsx) - pad)
        icy1 = max(0, int(min(ys) * self._fsy) - pad)
        icx2 = min(fw, int(max(xs) * self._fsx) + pad)
        icy2 = min(fh, int(max(ys) * self._fsy) + pad)
        inf_crop = frame[icy1:icy2, icx1:icx2]

        results = self._model(inf_crop, conf=CUP_CONF, verbose=False)
        boxes   = results[0].boxes

        cup_boxes_disp: list[tuple[int,int,int,int]] = []
        if boxes is not None and len(boxes) > 0:
            roi_poly = np.array(
                [[int(p[0] * self._fsx), int(p[1] * self._fsy)] for p in self.roi],
                dtype=np.float32)
            for xyxy, conf_val in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.conf.cpu().numpy()):
                rx1, ry1, rx2, ry2 = map(int, xyxy)
                fx1, fy1 = rx1 + icx1, ry1 + icy1
                fx2, fy2 = rx2 + icx1, ry2 + icy1
                fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                if cv2.pointPolygonTest(roi_poly, (fcx, fcy), False) < 0:
                    continue
                if conf_val < CUP_CONF:
                    continue
                bx1 = int(fx1 / self._fsx)
                by1 = int(fy1 / self._fsy)
                bx2 = int(fx2 / self._fsx)
                by2 = int(fy2 / self._fsy)
                cup_boxes_disp.append((bx1, by1, bx2, by2))

        cup_detected = len(cup_boxes_disp) > 0

        # Keep last known box for display stability when cup is confirmed
        if not cup_detected and self.cup_start is not None:
            cup_boxes_disp = self._last_boxes

        # State machine
        if cup_detected:
            if (self.cup_gone_since is not None and
                    self.cup_start is not None and
                    now_ts - self.cup_gone_since >= 2.0):
                # Gone >2s then detected again → new cup, log the old one first
                self._write_log(self.cup_start, datetime.now())
                self.cup_start          = None
                self.cup_gone_since     = None
                self._cup_confirm_since = now_ts
            else:
                self.cup_gone_since = None  # brief flicker, cancel grace
                if self._cup_confirm_since is None:
                    self._cup_confirm_since = now_ts
                elif (self.cup_start is None and
                      now_ts - self._cup_confirm_since >= CUP_CONFIRM_SEC):
                    self.cup_start = datetime.now()
                    print(f"[Coffee] Cup placed at {self.cup_start.strftime('%H:%M:%S')}")
        else:
            self._cup_confirm_since = None
            if self.cup_start is not None:
                if self.cup_gone_since is None:
                    self.cup_gone_since = now_ts
                elif now_ts - self.cup_gone_since >= CUP_GRACE_SEC:
                    self._write_log(self.cup_start, datetime.now())
                    self.cup_start      = None
                    self.cup_gone_since = None

        self._last_boxes = cup_boxes_disp
        return cup_boxes_disp

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_overlay(self, display: np.ndarray,
                     cup_boxes: list[tuple[int,int,int,int]] | None = None):
        if len(self.roi) < 2:
            return
        pts   = np.array(self.roi, dtype=np.int32)
        color = (0, 255, 0) if cup_boxes else (0, 165, 255)
        cv2.polylines(display, [pts], isClosed=(len(self.roi) == 4),
                      color=color, thickness=2)
        if self.roi_confirmed and len(self.roi) >= 3:
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.12, display, 0.88, 0, display)
        for p in self.roi:
            cv2.circle(display, p, 5, color, -1)

        if self._model is None:
            status      = "No model"
            badge_color = (0, 0, 200)
        elif cup_boxes:
            status      = "CUP"
            badge_color = (0, 220, 0)
        else:
            status      = "Empty"
            badge_color = (0, 165, 255)

        (lw, lh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        dh, dw = display.shape[:2]
        lx = max(5, min(min(p[0] for p in self.roi), dw - lw - 12))
        ly = max(lh + 8, min(p[1] for p in self.roi) - 6)
        cv2.rectangle(display, (lx - 5, ly - lh - 5), (lx + lw + 5, ly + 5),
                      (0, 0, 0), -1)
        cv2.putText(display, status, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2)

        for (bx1, by1, bx2, by2) in (cup_boxes or []):
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 220, 0), 2)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _write_log(self, start: datetime, end: datetime):
        duration = round((end - start).total_seconds(), 1)
        if duration < 3.0:
            return
        row = [self.camera_name,
               start.strftime("%Y-%m-%d %H:%M:%S"),
               end.strftime("%Y-%m-%d %H:%M:%S"),
               duration]
        with _log_lock:
            with open(COFFEE_LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
        print(f"[COFFEE] {start.strftime('%H:%M:%S')} → "
              f"{end.strftime('%H:%M:%S')} ({duration}s)")
