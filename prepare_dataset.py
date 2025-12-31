import json
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm
import yaml

# ================= CONFIG =================

DATASET_DIR = Path("Datasets")      # مجلد الصور + JSON
YOLO_DIR = Path("yolo_dataset")     # مجلد إخراج YOLO

CLASS_MAP = {
    "tank": 0,                      # غيّر الاسم لو الكلاس مختلف في Labelme
}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ==========================================


def make_dirs():
    subdirs = [
        "images/all",
        "labels/all",
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]
    for sub in subdirs:
        (YOLO_DIR / sub).mkdir(parents=True, exist_ok=True)


def convert_labelme_to_yolo_seg():
    json_files = list(DATASET_DIR.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} JSON files in {DATASET_DIR.resolve()}")

    copied_images = 0

    for jp in tqdm(json_files, desc="Converting JSON → YOLO-Seg"):
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ناخذ فقط اسم الملف بدون المسار في imagePath
        image_name_raw = data.get("imagePath", jp.stem + ".jpg")
        image_name = Path(image_name_raw).name
        img_path = DATASET_DIR / image_name

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image for JSON '{jp.name}': tried {img_path}")
            continue

        h, w = img.shape[:2]

        dst_img_path = YOLO_DIR / "images/all" / image_name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)
            copied_images += 1

        lines = []

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in CLASS_MAP:
                continue

            cls_id = CLASS_MAP[label]
            pts = shape.get("points", [])

            if len(pts) < 3:
                continue

            seg = []
            for x, y in pts:
                seg.append(x / w)
                seg.append(y / h)

            if len(seg) >= 6:
                line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in seg)
                lines.append(line)

        label_output = YOLO_DIR / "labels/all" / f"{jp.stem}.txt"
        with open(label_output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"[INFO] Copied {copied_images} images into yolo_dataset/images/all")


def split_dataset():
    all_img_dir = YOLO_DIR / "images/all"
    all_lbl_dir = YOLO_DIR / "labels/all"

    images = sorted(
        list(all_img_dir.glob("*.jpg")) +
        list(all_img_dir.glob("*.jpeg")) +
        list(all_img_dir.glob("*.png"))
    )

    print(f"[INFO] Found {len(images)} images in {all_img_dir.resolve()} to split")

    if not images:
        print("[ERROR] No images found in 'images/all'. Cannot split dataset.")
        return

    random.seed(42)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    def move_pairs(img_list, split):
        moved = 0
        for img_path in img_list:
            lbl_path = all_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                print(f"[WARN] No label for image {img_path.name}")
                continue

            shutil.copy2(img_path, YOLO_DIR / f"images/{split}" / img_path.name)
            shutil.copy2(lbl_path, YOLO_DIR / f"labels/{split}" / lbl_path.name)
            moved += 1
        print(f"[INFO] Moved {moved} images to {split}")

    move_pairs(train_imgs, "train")
    move_pairs(val_imgs, "val")
    move_pairs(test_imgs, "test")


def write_data_yaml():
    max_id = max(CLASS_MAP.values())
    names = [""] * (max_id + 1)

    for k, v in CLASS_MAP.items():
        names[v] = k

    yaml_dict = {
        "path": "./yolo_dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }

    with open("data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)

    print(f"[INFO] data.yaml written to {Path('data.yaml').resolve()}")


def main():
    print("[STEP 1] Creating YOLO directories...")
    make_dirs()

    print("[STEP 2] Converting Labelme JSON → YOLOv8 Segmentation...")
    convert_labelme_to_yolo_seg()

    print("[STEP 3] Splitting dataset...")
    split_dataset()

    print("[STEP 4] Writing data.yaml...")
    write_data_yaml()

    print("\n✅ All done! YOLOv8 segmentation dataset ready in 'yolo_dataset/'")
    print("   You can now run: python train_yolo.py")


if __name__ == "__main__":
    main()
