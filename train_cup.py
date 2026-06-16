"""
Train YOLO cup detector from labeled dataset.
Run: python train_cup.py
"""
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path("dataset")
TRAIN_DIR   = Path("dataset_split/train")
VAL_DIR     = Path("dataset_split/val")
YAML_PATH   = Path("cup_dataset.yaml")
VAL_RATIO   = 0.2
EPOCHS      = 100
MODEL_BASE  = "yolo11s.pt"
PROJECT     = "runs/detect/cup_model"
NAME        = "cpu_v1"


def split_dataset():
    images = sorted((DATASET_DIR / "images").glob("*.jpg"))
    random.seed(42)
    random.shuffle(images)

    n_val   = max(1, int(len(images) * VAL_RATIO))
    val_set = set(img.stem for img in images[:n_val])

    for split, stems in [("train", [i.stem for i in images[n_val:]]),
                         ("val",   [i.stem for i in images[:n_val]])]:
        (DATASET_DIR.parent / f"dataset_split/{split}/images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR.parent / f"dataset_split/{split}/labels").mkdir(parents=True, exist_ok=True)
        for stem in stems:
            src_img = DATASET_DIR / "images" / f"{stem}.jpg"
            src_lbl = DATASET_DIR / "labels" / f"{stem}.txt"
            shutil.copy(src_img, DATASET_DIR.parent / f"dataset_split/{split}/images/{stem}.jpg")
            if src_lbl.exists():
                shutil.copy(src_lbl, DATASET_DIR.parent / f"dataset_split/{split}/labels/{stem}.txt")
            else:
                (DATASET_DIR.parent / f"dataset_split/{split}/labels/{stem}.txt").write_text("")

    n_train = len(images) - n_val
    print(f"Split: {n_train} train  /  {n_val} val")
    return n_train, n_val


def write_yaml():
    abs_train = (TRAIN_DIR).resolve()
    abs_val   = (VAL_DIR).resolve()
    YAML_PATH.write_text(
        f"train: {abs_train}\n"
        f"val:   {abs_val}\n"
        f"nc: 1\n"
        f"names: ['cup']\n"
    )
    print(f"Dataset YAML → {YAML_PATH}")


def main():
    if TRAIN_DIR.exists():
        shutil.rmtree("dataset_split")
    n_train, n_val = split_dataset()
    write_yaml()

    model = YOLO(MODEL_BASE)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=640,
        batch=4,
        device="cpu",
        plots=True,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
    )
    best = Path(PROJECT) / NAME / "weights/best.pt"
    print(f"\nDone. Best weights → {best}")
    print(f"Update CUP_MODEL_PATH in coffee_tracker.py to: runs/detect/cup_model/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
