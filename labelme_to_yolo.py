"""
Convert labelme JSON annotations to YOLO format.
- Images with JSON → labels txt with bounding boxes
- Images without JSON → empty txt (negative sample)

Output structure:
    dataset/
        images/   *.jpg
        labels/   *.txt
"""
import json
from pathlib import Path

IMAGES_DIR = Path("training_images")
OUT_DIR    = Path("dataset")
LABELS     = ["cup"]   # index = class id


def shape_to_yolo(shape: dict, img_w: int, img_h: int) -> str | None:
    pts = shape["points"]
    if shape["shape_type"] == "rectangle":
        x1, y1 = pts[0]
        x2, y2 = pts[1]
    elif shape["shape_type"] == "polygon" and len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None

    label = shape.get("label", "")
    if label not in LABELS:
        print(f"  WARNING: unknown label '{label}' — skipped")
        return None

    cls = LABELS.index(label)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = abs(x2 - x1) / img_w
    h  = abs(y2 - y1) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    out_images = OUT_DIR / "images"
    out_labels = OUT_DIR / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(IMAGES_DIR.glob("*.jpg"))
    if not jpgs:
        print(f"No images found in {IMAGES_DIR}/")
        return

    with_ann = 0
    without_ann = 0

    for jpg in jpgs:
        json_path = jpg.with_suffix(".json")
        txt_path  = out_labels / (jpg.stem + ".txt")
        img_dst   = out_images / jpg.name

        # Copy image
        img_dst.write_bytes(jpg.read_bytes())

        if not json_path.exists():
            txt_path.write_text("")
            without_ann += 1
            continue

        data   = json.loads(json_path.read_text())
        img_w  = data["imageWidth"]
        img_h  = data["imageHeight"]
        lines  = []
        for shape in data.get("shapes", []):
            line = shape_to_yolo(shape, img_w, img_h)
            if line:
                lines.append(line)

        txt_path.write_text("\n".join(lines))
        with_ann += 1

    print(f"Done.")
    print(f"  With annotations : {with_ann}")
    print(f"  Empty (no cup)   : {without_ann}")
    print(f"  Output           : {OUT_DIR}/")


if __name__ == "__main__":
    main()
