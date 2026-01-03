import os
import cv2
import numpy as np

# ================= CONFIG =================
ORIGINAL_ROOT = "Data_prep"
CLEAN_ROOT = "Data_prep/Cleaned_data"
FINAL_ROOT = "final_dataset"

TARGET_SIZE = (720, 720)
CLASSES = ["BCC", "BKL", "MEL", "NV"]
# =========================================


def enhance(img):
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.3, blur, -0.3, 0)

    return img


def collect_images(folder):
    images = []
    for f in sorted(os.listdir(folder)):
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images


def run():
    os.makedirs(FINAL_ROOT, exist_ok=True)

    for cls in CLASSES:
        print(f"[INFO] Processing class: {cls}")

        final_class_dir = os.path.join(FINAL_ROOT, cls)
        os.makedirs(final_class_dir, exist_ok=True)

        merged_images = []

        # -------- 1. Original images --------
        orig_dir = os.path.join(ORIGINAL_ROOT, cls)
        if os.path.exists(orig_dir):
            merged_images.extend(collect_images(orig_dir))

        # -------- 2. Cleaned images --------
        clean_dir = os.path.join(CLEAN_ROOT, f"{cls}_C")
        if os.path.exists(clean_dir):
            merged_images.extend(collect_images(clean_dir))

        print(f"    Total merged images: {len(merged_images)}")

        # -------- 3. Resize → Enhance → Rename --------
        for idx, img in enumerate(merged_images, start=1):
            img = cv2.resize(img, TARGET_SIZE)
            img = enhance(img)

            out_name = f"{cls}_{idx}.jpg"
            cv2.imwrite(
                os.path.join(final_class_dir, out_name),
                img
            )

    print("\n[✔] FINAL DATASET READY")


if __name__ == "__main__":
    run()
