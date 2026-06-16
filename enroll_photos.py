"""
Standalone enrollment script.
Put photos in photos/<name>/ then run:
    python enroll_photos.py

Detects the largest face in each photo, saves jpg+npy to
enrollment_previews/<name>/, and deletes the source file.
"""
import cv2
import numpy as np
from pathlib import Path

PHOTOS_DIR   = Path("photos")
PREVIEWS_DIR = Path("enrollment_previews")
MIN_DET_SCORE = 0.50
MIN_FACE_PX   = 35


def load_model():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        allowed_modules=["detection", "recognition"],
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def next_index(person_dir: Path) -> int:
    existing = [int(p.stem) for p in person_dir.glob("*.npy") if p.stem.isdigit()]
    return max(existing, default=0) + 1


def process_photos(app):
    if not PHOTOS_DIR.exists():
        print(f"photos/ folder not found — create it and add subfolders per person.")
        return

    added = 0
    skipped = 0

    for person_dir in sorted(PHOTOS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        images = [p for p in sorted(person_dir.iterdir())
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            continue

        print(f"\n[{name}] {len(images)} image(s) found")
        out_dir = PREVIEWS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ✗ {img_path.name} — could not read")
                skipped += 1
                continue

            try:
                faces = app.get(img)
            except Exception as e:
                print(f"  ✗ {img_path.name} — detection error: {e}")
                skipped += 1
                continue

            good = [f for f in faces
                    if f.det_score >= MIN_DET_SCORE
                    and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX]

            if not good:
                print(f"  ✗ {img_path.name} — no face detected")
                skipped += 1
                continue

            # take the largest face (by area)
            face = max(good, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding.copy()
            x1, y1, x2, y2 = face.bbox.astype(int)
            pad = 10
            h, w = img.shape[:2]
            crop = img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

            idx = next_index(out_dir)
            np.save(str(out_dir / f"{idx}.npy"), emb)
            cv2.imwrite(str(out_dir / f"{idx}.jpg"), crop)
            img_path.unlink()
            print(f"  ✓ {img_path.name} → enrollment_previews/{name}/{idx}.jpg")
            added += 1

    print(f"\nDone. {added} enrolled, {skipped} skipped.")
    summary = {}
    if PREVIEWS_DIR.exists():
        for d in sorted(PREVIEWS_DIR.iterdir()):
            if d.is_dir():
                count = len(list(d.glob("*.npy")))
                if count:
                    summary[d.name] = count
    print(f"Current enrollment: {summary}")


if __name__ == "__main__":
    print("Loading InsightFace...")
    app = load_model()
    print("Ready.\n")
    process_photos(app)
