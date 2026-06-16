"""
Face Re-Identification - background thread + 1s voting window.
Enrollment via: photos/ folder, interactive 'E' key, or rebuild from previews.

Folder structure for bulk enrollment:
    photos/
        kaemon/
            1.jpg
            2.jpg
        pengyu/
            1.jpg
"""
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, deque

SIMILARITY_THRESH = 0.55
MARGIN_THRESH     = 0.10   # second-best must be this far below best
TOP_K             = 5      # mean of best-K embeddings per person (fairer across dataset sizes)
MIN_DET_SCORE     = 0.50
MIN_FACE_PX       = 35
VOTE_WINDOW_SEC   = 1.0    # rolling window for first confirmation
CHANGE_WINDOW_SEC = 2.0    # longer window required to override a confirmed name
CHANGE_THRESH     = 0.90   # 90% majority needed to change an already-confirmed name

PHOTOS_DIR   = Path("photos")
PREVIEWS_DIR = Path("enrollment_previews")


class FaceReID:
    """
    Stable named identity from face embeddings.

    submit(tid, crop)          — queue latest crop for background processing
    get_name(tid)              → (name | None, prev_name | None)
                                 prev_name is non-None only on first call after a change
    remove_track(tid)          — call when BoT-SORT track permanently gone
    """

    def __init__(self):
        print("[FaceReID] Loading InsightFace ArcFace...")
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                allowed_modules=["detection", "recognition"],
                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            print("[FaceReID] Ready.")
        except ImportError:
            print("[FaceReID] WARNING: pip install insightface onnxruntime")
            self._app = None

        self._enrolled:  dict[str, list[np.ndarray]] = {}
        self._emb_matrix: dict[str, np.ndarray] = {}  # name → (N,512) for fast matmul

        # Per-track vote history: tid → deque of (timestamp, name)
        self._votes:          dict[int, deque]      = {}
        # Confirmed name per track (only set after majority vote)
        self._conf_name:      dict[int, str]        = {}
        # Last name returned to caller (for change detection)
        self._last_name:      dict[int, str | None] = {}
        # When the last successful face vote was cast per track
        self._last_vote_time: dict[int, float]      = {}

        self._state_lock = threading.Lock()

        # Latest unprocessed crop per track (replaced each frame, never queues)
        self._pending:      dict[int, np.ndarray] = {}
        self._pending_lock  = threading.Lock()

        self._load_previews()
        self._load_photos_dir()

        threading.Thread(target=self._bg_loop, daemon=True).start()

    # ── public API ────────────────────────────────────────────────────────────

    def submit(self, tid: int, crop: np.ndarray) -> None:
        """Queue latest crop for this track. Non-blocking — always replaces previous."""
        if self._app is None or not self._enrolled or crop is None or crop.size == 0:
            return
        with self._pending_lock:
            self._pending[tid] = crop.copy()

    def get_name(self, tid: int) -> tuple[str | None, str | None]:
        """
        Returns (confirmed_name, prev_name).
        - confirmed_name: stable majority-voted name, or None if unrecognised
        - prev_name: the previous name if it just changed, else None
        """
        with self._state_lock:
            # If no face vote received in 2× the voting window, revert to unknown
            if tid in self._conf_name:
                stale = time.time() - self._last_vote_time.get(tid, 0) > VOTE_WINDOW_SEC * 2
                if stale:
                    self._conf_name.pop(tid)

            name = self._conf_name.get(tid)
            last = self._last_name.get(tid)
            self._last_name[tid] = name
            if last != name:
                return name, last
            return name, None

    def remove_track(self, tid: int) -> None:
        with self._state_lock:
            self._votes.pop(tid, None)
            self._conf_name.pop(tid, None)
            self._last_name.pop(tid, None)
            self._last_vote_time.pop(tid, None)
        with self._pending_lock:
            self._pending.pop(tid, None)

    # ── enrollment helpers (called from UI / photos loader) ───────────────────

    def detect_for_display(self, frame: np.ndarray) -> list[tuple]:
        if self._app is None or frame is None:
            return []
        try:
            faces = self._app.get(frame)
            return [tuple(f.bbox.astype(int)) for f in faces if f.det_score >= MIN_DET_SCORE]
        except Exception:
            return []

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """Return list of {bbox, embedding, crop} for all detected faces."""
        if self._app is None or frame is None:
            return []
        try:
            faces = self._app.get(frame)
        except Exception:
            return []
        result = []
        for f in faces:
            if f.det_score < MIN_DET_SCORE:
                continue
            x1, y1, x2, y2 = f.bbox.astype(int)
            if (y2 - y1) < MIN_FACE_PX:
                continue
            pad = 10
            h, w = frame.shape[:2]
            crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
            result.append({
                "bbox":      (x1, y1, x2, y2),
                "embedding": f.normed_embedding.copy(),
                "crop":      crop,
            })
        return result

    def _rebuild_matrix(self, name: str):
        embs = self._enrolled.get(name)
        if embs:
            self._emb_matrix[name] = np.stack(embs)  # (N, 512)

    def enroll_embedding(self, name: str, embedding: np.ndarray, face_img: np.ndarray | None = None):
        """Enroll a pre-computed embedding. Saves .npy (and optionally .jpg) to previews folder."""
        self._enrolled.setdefault(name, []).append(embedding)
        self._rebuild_matrix(name)
        count = len(self._enrolled[name])
        preview_dir = PREVIEWS_DIR / name
        preview_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(preview_dir / f"{count}.npy"), embedding)
        if face_img is not None:
            cv2.imwrite(str(preview_dir / f"{count}.jpg"), face_img)
        print(f"[FaceReID] Enrolled '{name}' (photo #{count})")

    def get_enrolled(self) -> dict[str, int]:
        return {name: len(embs) for name, embs in self._enrolled.items()}

    def rebuild_from_previews(self):
        """Resync enrolled embeddings from enrollment_previews/ folder."""
        if not PREVIEWS_DIR.exists():
            print("[FaceReID] No previews folder found.")
            return
        new_enrolled: dict[str, list] = {}
        for person_dir in sorted(PREVIEWS_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            npy_files = sorted(person_dir.glob("*.npy"))
            if npy_files:
                for npy_path in npy_files:
                    jpg_path = npy_path.with_suffix(".jpg")
                    if not jpg_path.exists():
                        print(f"[FaceReID] {npy_path.name} has no jpg — skipped")
                        continue
                    emb = np.load(str(npy_path))
                    new_enrolled.setdefault(name, []).append(emb)
            else:
                for img_path in sorted(person_dir.glob("*.jpg")):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    ph, pw = img.shape[:2]
                    pad = max(ph, pw)
                    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                                cv2.BORDER_CONSTANT, value=(128, 128, 128))
                    emb = self._embed_image(padded)
                    if emb is not None:
                        new_enrolled.setdefault(name, []).append(emb)
                        np.save(str(img_path.with_suffix(".npy")), emb)
                    else:
                        print(f"[FaceReID] No face in {img_path.name} — skipped")
        if not new_enrolled:
            print("[FaceReID] Rebuild found nothing — keeping existing enrollment unchanged.")
            return
        self._enrolled = new_enrolled
        self._emb_matrix = {}
        for name in self._enrolled:
            self._rebuild_matrix(name)
        print(f"[FaceReID] Rebuilt: { {n: len(e) for n, e in self._enrolled.items()} }")

    # ── background thread ─────────────────────────────────────────────────────

    def _bg_loop(self) -> None:
        while True:
            with self._pending_lock:
                batch = dict(self._pending)
                self._pending.clear()

            if not batch:
                time.sleep(0.02)
                continue

            for tid, crop in batch.items():
                name = self._detect_and_match(crop)
                if name is not None:
                    self._cast_vote(tid, name)

    def _cast_vote(self, tid: int, name: str) -> None:
        now = time.time()
        with self._state_lock:
            self._last_vote_time[tid] = now
            current = self._conf_name.get(tid)
            window = CHANGE_WINDOW_SEC if current is not None else VOTE_WINDOW_SEC
            q = self._votes.setdefault(tid, deque())
            q.append((now, name))
            while q and (now - q[0][0]) > window:
                q.popleft()
            if len(q) < 2:
                return
            counts = Counter(n for _, n in q)
            top, cnt = counts.most_common(1)[0]
            threshold = CHANGE_THRESH if (current is not None and current != top) else 0.51
            if cnt / len(q) >= threshold:
                self._conf_name[tid] = top

    # ── detection & matching ──────────────────────────────────────────────────

    def _detect_and_match(self, crop: np.ndarray) -> str | None:
        try:
            faces = self._app.get(crop)
        except Exception:
            return None
        if not faces:
            return None
        good = [f for f in faces if f.det_score >= MIN_DET_SCORE
                and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX]
        if not good:
            return None
        h_crop, w_crop = crop.shape[:2]
        cx_img, cy_img = w_crop / 2.0, h_crop / 2.0
        face = min(good, key=lambda f: (
            ((f.bbox[0]+f.bbox[2])/2 - cx_img)**2 +
            ((f.bbox[1]+f.bbox[3])/2 - cy_img)**2
        ))

        emb = face.normed_embedding

        scores = {}
        for name, mat in self._emb_matrix.items():
            sims = mat @ emb                          # (N,) — one matmul instead of N dot calls
            top_k = np.partition(sims, -min(TOP_K, len(sims)))[-min(TOP_K, len(sims)):]
            scores[name] = float(top_k.mean())
        if not scores:
            return None
        sorted_sims = sorted(scores.values(), reverse=True)
        best_name = max(scores, key=scores.get)
        best_sim  = sorted_sims[0]

        if best_sim < SIMILARITY_THRESH:
            return None
        if len(sorted_sims) > 1 and (best_sim - sorted_sims[1]) < MARGIN_THRESH:
            return None
        return best_name

    def _embed_image(self, img: np.ndarray) -> np.ndarray | None:
        try:
            faces = self._app.get(img)
        except Exception:
            return None
        good = [f for f in faces if f.det_score >= MIN_DET_SCORE
                and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX]
        if not good:
            return None
        return max(good, key=lambda f: f.det_score).normed_embedding.copy()

    def _load_previews(self):
        if not PREVIEWS_DIR.exists():
            print("[FaceReID] No enrolled faces. Press E to enroll.")
            return
        for person_dir in sorted(PREVIEWS_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for npy_path in sorted(person_dir.glob("*.npy")):
                jpg_path = npy_path.with_suffix(".jpg")
                if not jpg_path.exists():
                    continue
                emb = np.load(str(npy_path))
                self._enrolled.setdefault(name, []).append(emb)
        for name in self._enrolled:
            self._rebuild_matrix(name)
        if self._enrolled:
            print(f"[FaceReID] Loaded enrolled: { {n: len(e) for n, e in self._enrolled.items()} }")
        else:
            print("[FaceReID] No enrolled faces. Press E to enroll.")

    def _load_photos_dir(self):
        if self._app is None or not PHOTOS_DIR.exists():
            return
        added = 0
        for person_dir in sorted(PHOTOS_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for img_path in sorted(person_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                try:
                    faces = self._app.get(img)
                except Exception:
                    continue
                good = [f for f in faces if f.det_score >= MIN_DET_SCORE
                        and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX]
                if not good:
                    print(f"[FaceReID] No face in {img_path.name} — skipped")
                    continue
                face = max(good, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = face.normed_embedding.copy()
                x1, y1, x2, y2 = face.bbox.astype(int)
                pad = 10
                h, w = img.shape[:2]
                crop = img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
                self.enroll_embedding(name, emb, crop)
                img_path.unlink()
                added += 1
        if added:
            print(f"[FaceReID] Loaded from photos/: { {n: len(e) for n, e in self._enrolled.items()} }")
