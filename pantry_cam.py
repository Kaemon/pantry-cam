import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"

import cv2
import csv
import time
import json
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

from coffee_tracker import CoffeeTracker, init_coffee_log
from face_tracker import FaceReID

load_dotenv()

MODEL_PATH   = "yolo11s.pt"
CONF         = 0.30
PERSON_CLASS = 0
ROI_CONFIG   = "pantry_roi.json"
LOG_FILE      = f"pantry_log_{datetime.now().strftime('%Y-%m-%d')}.csv"
CROPS_DIR     = Path("crops")

CAMERA_NAME  = "Pantry Cam"
CAMERA_URL   = os.getenv("RTSP_CAM1")
GRACE_SEC    = 15
MIN_STAY_SEC = 5.0
MIN_BOX_H    = 150
MIN_BOX_W    = 80
CROP_INTERVAL = 5.0
GHOST_SEC    = 0.5

yolo_model = YOLO(MODEL_PATH)

# ── Crop saving ───────────────────────────────────────────────────────────────

_crop_buffer:    dict[int, list[np.ndarray]] = {}
_last_crop_time: dict[int, float]            = {}
_last_track_box: dict[int, tuple]            = {}


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)



def _save_crop(pid: int, crop: np.ndarray):
    if crop.size == 0:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    _crop_buffer.setdefault(pid, []).append((ts, crop.copy()))


def _flush_crop(pid: int):
    crops = _crop_buffer.pop(pid, [])
    _last_crop_time.pop(pid, None)
    if not crops:
        return
    person_dir = CROPS_DIR / f"person_{pid:03d}"
    person_dir.mkdir(parents=True, exist_ok=True)
    for ts, crop in crops:
        cv2.imwrite(str(person_dir / f"{ts}.jpg"), crop)


# ── Pantry ROI ────────────────────────────────────────────────────────────────

roi_points: list[tuple[int, int]] = []
roi_confirmed = False


def mouse_cb(event, x, y, flags, param):
    global roi_points, roi_confirmed
    if roi_confirmed:
        return
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 8:
        roi_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
        roi_points.pop()


def save_roi():
    with open(ROI_CONFIG, "w") as f:
        json.dump({"points": roi_points}, f)
    print(f"ROI saved → {ROI_CONFIG}")


def load_roi() -> bool:
    global roi_points, roi_confirmed
    if Path(ROI_CONFIG).exists():
        data = json.load(open(ROI_CONFIG))
        roi_points = [tuple(p) for p in data["points"]]
        roi_confirmed = True
        print(f"Loaded ROI: {roi_points}")
        return True
    return False


def point_in_roi(pt: tuple[int, int]) -> bool:
    if len(roi_points) < 3:
        return False
    poly = np.array(roi_points, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def draw_roi_overlay(frame: np.ndarray):
    if len(roi_points) < 2:
        return
    pts = np.array(roi_points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=(len(roi_points) >= 3),
                  color=(0, 255, 255), thickness=2)
    if roi_confirmed and len(roi_points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    for i, p in enumerate(roi_points):
        cv2.circle(frame, p, 5, (0, 200, 255), -1)
        cv2.putText(frame, str(i + 1), (p[0] + 7, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


# ── Logging ───────────────────────────────────────────────────────────────────

log_lock = threading.Lock()
session_total = 0


def init_log():
    if not Path(LOG_FILE).exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["camera", "track_id", "entry_time", "exit_time", "duration_sec"]
            )


def write_log(tid: int, name: str | None, entry: datetime, exit_: datetime):
    global session_total
    duration = round((exit_ - entry).total_seconds(), 1)
    if duration < MIN_STAY_SEC:
        _crop_buffer.pop(tid, None)
        return
    session_total += 1
    _flush_crop(tid)
    label = name if name else f"#{tid}"
    row = [CAMERA_NAME, label,
           entry.strftime("%Y-%m-%d %H:%M:%S"),
           exit_.strftime("%Y-%m-%d %H:%M:%S"),
           duration]
    with log_lock:
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)
    print(f"[LOG] {label:<14}  {entry.strftime('%H:%M:%S')} → "
          f"{exit_.strftime('%H:%M:%S')}  ({duration}s)")


# ── Camera stream ─────────────────────────────────────────────────────────────

class CamStream:
    def __init__(self, url: str):
        self.url = url
        self.frame: np.ndarray | None = None
        self.connected = False
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        cap = self._open()
        while self.running:
            if cap is None:
                time.sleep(5)
                cap = self._open()
                continue
            ret = cap.grab()
            if not ret:
                print(f"[{CAMERA_NAME}] Connection lost, retrying in 5s...")
                cap.release()
                cap = None
                self.connected = False
                continue
            ret, frame = cap.retrieve()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.connected = True

    def _open(self):
        if not self.url:
            print(f"[{CAMERA_NAME}] No URL configured")
            return None
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print(f"[{CAMERA_NAME}] Connected to {self.url}")
            return cap
        print(f"[{CAMERA_NAME}] Failed to connect")
        return None

    def get_frame(self) -> np.ndarray | None:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


# ── ROI setup (interactive) ───────────────────────────────────────────────────

def run_roi_setup(cam: CamStream):
    global roi_confirmed
    win = "Pantry ROI Setup"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)

    print("\n=== ROI Setup ===")
    print("Click 4 corners of the pantry area (in order).")
    print("Right-click to undo last point.")
    print("Press ENTER to confirm  |  R to reset\n")

    while True:
        frame = cam.get_frame()
        if frame is None:
            ph = np.zeros((540, 960, 3), np.uint8)
            cv2.putText(ph, "Waiting for camera...", (30, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(win, ph)
            cv2.waitKey(100)
            continue

        display = cv2.resize(frame, (960, 540))
        draw_roi_overlay(display)
        hints = [
            f"Points: {len(roi_points)}/8  — left-click to add, right-click to undo",
            "Press ENTER to confirm  |  R to reset",
        ]
        for i, txt in enumerate(hints):
            cv2.putText(display, txt, (10, 510 - (len(hints) - 1 - i) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(roi_points) >= 4:
            roi_confirmed = True
            save_roi()
            cv2.destroyWindow(win)
            return
        elif key == ord('r'):
            roi_points.clear()


# ── Tracking loop ─────────────────────────────────────────────────────────────

def run_tracking(cam: CamStream, coffee: CoffeeTracker, reid: FaceReID):
    global roi_confirmed

    in_zone:      dict[int, datetime]               = {}
    pending_exit: dict[int, tuple[datetime,datetime]] = {}
    tid_names:    dict[int, str]                     = {}
    ghost_boxes:  dict[int, tuple]                   = {}
    ghost_ts:     dict[int, float]                   = {}

    # Enrollment state (same window as tracking)
    enroll_active  = False
    en_state       = "idle"   # idle | selecting | naming | feedback
    en_face_boxes  = []
    en_last_detect = 0.0
    en_captured    = None
    en_faces       = []
    en_sel         = -1
    en_name        = ""
    en_msg         = ""
    en_msg_color   = (0, 255, 0)
    en_msg_ts      = 0.0

    print("\nTracking started. Press Q to quit  |  R to redraw ROI  |  C to set coffee ROI  |  E to enroll\n")

    while True:
        frame = cam.get_frame()

        if frame is None:
            ph = np.zeros((540, 960, 3), np.uint8)
            cv2.putText(ph, f"{CAMERA_NAME} — OFFLINE", (240, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            cv2.imshow(CAMERA_NAME, ph)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break
            continue

        h, w = frame.shape[:2]
        sx, sy = 960 / w, 540 / h

        # ── Enrollment mode (skip YOLO entirely for instant key response) ──────
        if enroll_active:
            esx, esy = 960 / w, 540 / h
            if en_state == "idle":
                ed = cv2.resize(frame, (960, 540))
                now_t = time.time()
                if now_t - en_last_detect > 0.5:
                    en_face_boxes = reid.detect_for_display(frame)
                    en_last_detect = now_t
                for i, (fx1, fy1, fx2, fy2) in enumerate(en_face_boxes):
                    cv2.rectangle(ed, (int(fx1*esx), int(fy1*esy)),
                                  (int(fx2*esx), int(fy2*esy)), (0, 255, 0), 2)
                    cv2.putText(ed, str(i+1), (int(fx1*esx)+4, int(fy1*esy)+24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                enrolled_str = ", ".join(f"{n}({c})" for n,c in reid.get_enrolled().items()) or "none"
                n_f = len(en_face_boxes)
                for i, txt in enumerate([
                    "ENROLLMENT  (SPACE=capture, number=select, ENTER=save)",
                    f"Enrolled: {enrolled_str}",
                    f"{n_f} face(s) detected — {'ready' if n_f else 'move closer'}",
                    "B=rebuild   ESC=back to tracking",
                ]):
                    cv2.putText(ed, txt, (20, 40+i*32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (0,255,0) if i==2 and n_f else (0,255,255), 2)
            elif en_state == "selecting":
                ed = cv2.resize(en_captured, (960, 540))
                for i, face in enumerate(en_faces):
                    fx1,fy1,fx2,fy2 = face["bbox"]
                    cv2.rectangle(ed,(int(fx1*esx),int(fy1*esy)),(int(fx2*esx),int(fy2*esy)),(0,255,255),2)
                    cv2.putText(ed,str(i+1),(int(fx1*esx)+4,int(fy1*esy)+28),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
                cv2.putText(ed,f"Press 1-{len(en_faces)} to select   ESC=back",(20,510),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
            elif en_state == "naming":
                ed = cv2.resize(en_captured, (960, 540))
                fx1,fy1,fx2,fy2 = en_faces[en_sel]["bbox"]
                cv2.rectangle(ed,(int(fx1*esx),int(fy1*esy)),(int(fx2*esx),int(fy2*esy)),(0,255,0),3)
                cv2.putText(ed,"Type name + ENTER   ESC=back",(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,200,255),2)
                cv2.putText(ed,f"> {en_name}_",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
            elif en_state == "feedback":
                src = en_captured if en_captured is not None else frame
                ed = cv2.resize(src, (960, 540))
                cv2.putText(ed, en_msg, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, en_msg_color, 2)
                if time.time() - en_msg_ts > 1.5:
                    en_state = "selecting" if len(en_faces) > 1 else "idle"
            cv2.imshow(CAMERA_NAME, ed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if en_state == "idle":
                if key == 27:
                    enroll_active = False; en_face_boxes = []
                elif key == ord('b'):
                    reid.rebuild_from_previews()
                    en_msg = f"Rebuilt: {reid.get_enrolled()}"
                    en_msg_color = (0, 255, 0); en_msg_ts = time.time(); en_state = "feedback"
                elif key == ord(' '):
                    en_captured = frame.copy()
                    en_faces = reid.detect_faces(en_captured)
                    if not en_faces:
                        en_msg = "No face detected — try again"; en_msg_color = (0,80,255)
                        en_msg_ts = time.time(); en_state = "feedback"; en_captured = None
                    elif len(en_faces) == 1:
                        en_sel = 0; en_name = ""; en_state = "naming"
                    else:
                        en_state = "selecting"
            elif en_state == "selecting":
                if key == 27:
                    en_state = "idle"
                elif ord('1') <= key <= ord('1') + len(en_faces) - 1:
                    en_sel = key - ord('1'); en_name = ""; en_state = "naming"
            elif en_state == "naming":
                if key == 27:
                    en_state = "selecting" if len(en_faces) > 1 else "idle"
                elif key == 13 and en_name.strip():
                    face = en_faces[en_sel]
                    reid.enroll_embedding(en_name.strip(), face["embedding"], face.get("crop"))
                    en_msg = f"Enrolled '{en_name.strip()}'"
                    en_msg_color = (0,255,0); en_msg_ts = time.time(); en_state = "feedback"; en_name = ""
                elif key == 8:
                    en_name = en_name[:-1]
                elif 32 <= key <= 126:
                    en_name += chr(key)
            continue  # skip YOLO tracking

        # Primary: YOLO + BoT-SORT (downsample for speed)
        small = cv2.resize(frame, (640, 360))
        results = yolo_model.track(
            small, classes=[PERSON_CLASS], conf=CONF,
            iou=0.7, persist=True, tracker="botsort_pantry.yaml", verbose=False
        )
        tracked: dict[int, tuple] = {}
        bx_scale, by_scale = w / 640, h / 360
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for tid_f, box in zip(results[0].boxes.id.cpu().numpy(),
                                  results[0].boxes.xyxy.cpu().numpy()):
                tid = int(tid_f)
                x1 = int(box[0] * bx_scale); y1 = int(box[1] * by_scale)
                x2 = int(box[2] * bx_scale); y2 = int(box[3] * by_scale)
                tracked[tid] = (x1, y1, x2, y2)

        _last_track_box.update(tracked)
        _ghost_now = time.time()
        for tid, box in tracked.items():
            ghost_boxes[tid] = box
            ghost_ts[tid] = _ghost_now

        current_tids: set[int] = set()
        person_boxes_frame: list[tuple[int,int,int,int]] = []

        for tid, (x1, y1, x2, y2) in tracked.items():
            if (y2 - y1) < MIN_BOX_H or (x2 - x1) < MIN_BOX_W:
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            inside = point_in_roi((int(cx * sx), int(cy * sy)))

            current_tids.add(tid)
            person_boxes_frame.append((x1, y1, x2, y2))

            # Submit crop every frame for background voting-based ReID
            pad = 10
            crop = frame[max(0, y1-pad):min(h, y2+pad),
                         max(0, x1-pad):min(w, x2+pad)]
            reid.submit(tid, crop)

            name, _ = reid.get_name(tid)
            if name and tid_names.get(tid) != name:
                tid_names[tid] = name
                # Merge any existing zone entries for the same name under a different track
                for src in [t for t in list(in_zone) if t != tid and tid_names.get(t) == name]:
                    old_entry = in_zone.pop(src)
                    in_zone[tid] = min(in_zone[tid], old_entry) if tid in in_zone else old_entry
                    tid_names.pop(src, None)
                    reid.remove_track(src)
                if tid in in_zone:
                    for src in [t for t in list(pending_exit) if t != tid and tid_names.get(t) == name]:
                        old_entry, _ = pending_exit.pop(src)
                        in_zone[tid] = min(in_zone[tid], old_entry)
                        tid_names.pop(src, None)
                        reid.remove_track(src)

            if inside:
                if tid in pending_exit:
                    entry_time, _ = pending_exit.pop(tid)
                    in_zone[tid] = entry_time
                if tid not in in_zone:
                    in_zone[tid] = datetime.now()
                    identified_name = tid_names.get(tid)
                    display = identified_name if identified_name else f"Person #{tid}"
                    if identified_name:
                        print(f"[FaceReID] Identified: {identified_name} (track #{tid})")
                    print(f"[{CAMERA_NAME}] {display} entered")
                now_ts = time.time()
                if now_ts - _last_crop_time.get(tid, 0) >= CROP_INTERVAL:
                    pad = 10
                    crop = frame[max(0, y1-pad):min(h, y2+pad),
                                 max(0, x1-pad):min(w, x2+pad)]
                    _save_crop(tid, crop)

            if not inside:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_name = tid_names.get(tid, f"#{tid}")
            label = display_name
            entry = in_zone.get(tid) or (pending_exit[tid][0] if tid in pending_exit else None)
            if entry:
                secs = int((datetime.now() - entry).total_seconds())
                label += f"  {secs}s"
            label_y = y1 + 20 if y1 < 20 else y1 - 7
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

        # Ghost boxes: display last known position for recently-missed tracks
        _ghost_now = time.time()
        for tid in [t for t in list(ghost_ts) if t not in tracked]:
            if _ghost_now - ghost_ts[tid] > GHOST_SEC:
                ghost_boxes.pop(tid, None)
                ghost_ts.pop(tid, None)
                continue
            x1, y1, x2, y2 = ghost_boxes[tid]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not point_in_roi((int(cx * sx), int(cy * sy))):
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
            display_name = tid_names.get(tid, f"#{tid}")
            label = display_name
            entry = in_zone.get(tid) or (pending_exit[tid][0] if tid in pending_exit else None)
            if entry:
                secs = int((datetime.now() - entry).total_seconds())
                label += f"  {secs}s"
            label_y = y1 + 20 if y1 < 20 else y1 - 7
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)

        # Grace-period exit logic
        now = datetime.now()
        for tid in [t for t in in_zone if t not in current_tids]:
            pending_exit[tid] = (in_zone.pop(tid), now)
        for tid in list(pending_exit):
            if tid in current_tids:
                pass
            elif (now - pending_exit[tid][1]).total_seconds() >= GRACE_SEC:
                entry_time, disappear_time = pending_exit.pop(tid)
                write_log(tid, tid_names.pop(tid, None), entry_time, disappear_time)
                reid.remove_track(tid)
                ghost_boxes.pop(tid, None)
                ghost_ts.pop(tid, None)

        # Cup detection (handled by CoffeeTracker)
        cup_boxes = coffee.detect(frame, person_boxes_frame)

        display = cv2.resize(frame, (960, 540))
        draw_roi_overlay(display)
        coffee.draw_overlay(display, cup_boxes)

        stats_txt = f"In pantry: {len(in_zone)}   Total: {session_total}"
        (sw, _), _ = cv2.getTextSize(stats_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(display, stats_txt, (950 - sw, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow(CAMERA_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi_points.clear()
            roi_confirmed = False
            run_roi_setup(cam)
        elif key == ord('c'):
            coffee.setup_roi(cam)
        elif key == ord('e'):
            enroll_active = True
            en_state = "idle"
        elif key == ord('p'):
            reid._load_photos_dir()
            print("[Main] photos/ folder reloaded.")

    now = datetime.now()
    for tid, entry in in_zone.items():
        write_log(tid, tid_names.get(tid), entry, now)
        reid.remove_track(tid)
    for tid, (entry, _) in pending_exit.items():
        write_log(tid, tid_names.get(tid), entry, now)
        reid.remove_track(tid)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    init_log()
    init_coffee_log()

    coffee = CoffeeTracker(CAMERA_NAME)
    coffee.load_roi()

    reid = FaceReID()

    cam = CamStream(CAMERA_URL)
    cam.start()

    print("Connecting to camera...")
    for _ in range(20):
        if cam.get_frame() is not None:
            break
        time.sleep(0.5)

    if not load_roi():
        if cam.get_frame() is None:
            print("ERROR: Camera not connected and no saved ROI. Exiting.")
            return
        run_roi_setup(cam)

    run_tracking(cam, coffee, reid)

    cam.running = False
    cv2.destroyAllWindows()
    print(f"\nDone. Log → {LOG_FILE}")


if __name__ == "__main__":
    main()
