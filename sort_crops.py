"""
Auto-sort crops/ into photos/<name>/ using existing enrollment data.

For each person_XXX/ folder:
  - Scans ALL frames; each frame picks the largest face if multiple faces present
  - Uses majority vote across all frames to decide identity
  - Confident majority  → copies all frames to photos/<name>/
  - Unknown/ambiguous   → copies to photos/unknown/person_XXX/

Then run enroll_photos.py (or press P in main window) to enroll them.

    python sort_crops.py
"""
import cv2
import shutil
import numpy as np
from collections import Counter
from pathlib import Path

CROPS_DIR     = Path("crops")
PHOTOS_DIR    = Path("photos")
PREVIEWS_DIR  = Path("enrollment_previews")

MATCH_THRESH  = 0.58   # per-frame match threshold
MARGIN_THRESH = 0.07   # gap between 1st and 2nd match per frame
VOTE_THRESH   = 0.50   # fraction of frames that must agree on the winner
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


def load_enrolled() -> dict[str, list[np.ndarray]]:
    enrolled = {}
    if not PREVIEWS_DIR.exists():
        return enrolled
    for person_dir in sorted(PREVIEWS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for npy_path in sorted(person_dir.glob("*.npy")):
            if not npy_path.with_suffix(".jpg").exists():
                continue
            emb = np.load(str(npy_path))
            enrolled.setdefault(name, []).append(emb)
    return enrolled


def detect_largest_face(app, img: np.ndarray):
    """Return embedding for the largest face in the image, or None."""
    try:
        faces = app.get(img)
    except Exception:
        return None
    good = [f for f in faces
            if f.det_score >= MIN_DET_SCORE
            and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX]
    if not good:
        return None
    face = max(good, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return face.normed_embedding.copy()


def match_embedding(emb: np.ndarray, enrolled: dict) -> tuple[str, float, float]:
    """Return (best_name, best_sim, margin)."""
    scores = {name: max(float(np.dot(emb, e)) for e in emb_list)
              for name, emb_list in enrolled.items()}
    sorted_sims = sorted(scores.values(), reverse=True)
    best_name = max(scores, key=scores.get)
    best_sim  = sorted_sims[0]
    margin    = (best_sim - sorted_sims[1]) if len(sorted_sims) > 1 else 1.0
    return best_name, best_sim, margin


def sort_crops(app, enrolled: dict):
    if not CROPS_DIR.exists():
        print("crops/ not found.")
        return

    person_dirs = [d for d in sorted(CROPS_DIR.iterdir()) if d.is_dir()]
    if not person_dirs:
        print("No person folders in crops/.")
        return

    if not enrolled:
        print("No enrollment data found. Run the main program and enroll people first.")
        return

    results = []

    for person_dir in person_dirs:
        images = sorted([p for p in person_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not images:
            continue

        votes: Counter = Counter()
        n_faces = 0
        print(f"  {person_dir.name}  ({len(images)} imgs)...", end=" ", flush=True)

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = detect_largest_face(app, img)
            if emb is None:
                continue
            n_faces += 1
            best_name, best_sim, margin = match_embedding(emb, enrolled)
            if best_sim >= MATCH_THRESH and margin >= MARGIN_THRESH:
                votes[best_name] += 1
            else:
                votes["__unknown__"] += 1

        if n_faces == 0:
            print("NO FACE")
            results.append((person_dir.name, None, 0, 0, 0, images))
            continue

        total = sum(votes.values())
        winner, win_count = votes.most_common(1)[0]

        if winner == "__unknown__" or win_count / total < VOTE_THRESH:
            named = [(n, c) for n, c in votes.most_common() if n != "__unknown__"]
            matched_name = named[0][0] if named else None
            named_votes = named[0][1] if named else 0
            unk_votes = votes.get("__unknown__", 0)
            print(f"unknown  (best named: {matched_name} {named_votes}/{total}, unclear: {unk_votes}/{total})")
            results.append((person_dir.name, matched_name, win_count, total, n_faces, images))
        else:
            print(f"→ {winner}  {win_count}/{total} ({win_count/total:.0%})")
            results.append((person_dir.name, winner, win_count, total, n_faces, images))

    # Print report
    print(f"\n{'Person':<15} {'Match':<14} {'Votes':>10}  {'Faces':>6}  Status")
    print("-" * 65)
    confirmed = 0
    ambiguous = 0
    unknown   = 0

    for row in results:
        person_name, matched_name, win_count, total, n_faces, images = row
        if n_faces == 0:
            status = "NO FACE"
            unknown += 1
            print(f"{person_name:<15} {'—':<14} {'—':>10}  {n_faces:>6}  {status}")
        elif matched_name is None or total == 0 or win_count / total < VOTE_THRESH:
            status = "unknown"
            unknown += 1
            print(f"{person_name:<15} {(matched_name or '—'):<14} {f'{win_count}/{total}':>10}  {n_faces:>6}  {status}")
        else:
            vote_frac = win_count / total
            if vote_frac >= VOTE_THRESH:
                status = "✓ matched"
                confirmed += 1
            else:
                status = "ambiguous"
                ambiguous += 1
            print(f"{person_name:<15} {matched_name:<14} {f'{win_count}/{total} ({vote_frac:.0%})':>10}  {n_faces:>6}  {status}")

    print(f"\nSummary: {confirmed} matched, {ambiguous} ambiguous, {unknown} unknown  (out of {len(results)})")

    print()
    ans = input("Copy matched photos to photos/<name>/  and unknowns to photos/unknown/? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled — no files copied.")
        return

    for row in results:
        person_name, matched_name, win_count, total, n_faces, images = row
        if n_faces == 0:
            continue

        vote_frac = win_count / total if total > 0 else 0
        if matched_name and vote_frac >= VOTE_THRESH:
            out_dir = PHOTOS_DIR / matched_name
        else:
            out_dir = PHOTOS_DIR / "unknown" / person_name

        out_dir.mkdir(parents=True, exist_ok=True)
        for src in images:
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))

    print("Done. Run  python enroll_photos.py  (or press P in main window) to enroll.")


if __name__ == "__main__":
    print("Loading InsightFace...")
    app = load_model()
    enrolled = load_enrolled()
    print(f"Loaded enrollment: { {n: len(e) for n, e in enrolled.items()} }\n")
    sort_crops(app, enrolled)
