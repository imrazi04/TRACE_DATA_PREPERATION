# hair_remove.py
import os
import cv2
import numpy as np

INPUT_ROOT = "Data_prep/Hairy_data"
OUTPUT_ROOT = "Data_prep/Cleaned_data"

TOPHAT_RADIUS = 18
BRIGHTENING_FACTOR = 0.7
FFC_SIGMA = 30


def process_channel(channel):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * TOPHAT_RADIUS + 1, 2 * TOPHAT_RADIUS + 1)
    )
    closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
    tophat = cv2.subtract(closed, channel)

    bright = tophat.astype(np.float32)
    bright += (255 - bright) * BRIGHTENING_FACTOR
    bright = np.clip(bright, 0, 255).astype(np.uint8)

    offset = cv2.GaussianBlur(bright.astype(np.float32), (0, 0), FFC_SIGMA)
    ffc = (bright / (offset + 1e-6)) * np.mean(offset)
    return np.clip(ffc, 0, 255).astype(np.uint8)


def remove_hair(img):
    processed = [
        process_channel(img[:, :, 0]),
        process_channel(img[:, :, 1]),
        process_channel(img[:, :, 2])
    ]
    enhanced = np.maximum.reduce(processed)

    _, mask = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return cv2.inpaint(img, (mask > 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)


def run():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for cls in ["BCC", "BKL", "MEL", "NV"]:
        src = os.path.join(INPUT_ROOT, f"{cls}_h")
        dst = os.path.join(OUTPUT_ROOT, f"{cls}_C")
        os.makedirs(dst, exist_ok=True)

        for i, f in enumerate(os.listdir(src)):
            img = cv2.imread(os.path.join(src, f))
            if img is None:
                continue

            clean = remove_hair(img)
            cv2.imwrite(
                os.path.join(dst, f"{cls}_clean_{i:05d}.jpg"),
                clean
            )


if __name__ == "__main__":
    run()
