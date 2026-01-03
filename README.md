# TRACE Data Preparation ✅

## Project overview
This repository prepares a cleaned and augmented dermoscopic image dataset based on ISIC 2019 for downstream lesion classification tasks. The goal is to remove common imaging artifacts (primarily hair), standardize image size, and address class imbalance so models trained on this data perform more robustly in real-world conditions.

## What has been done so far ✅
- Separated hairy and non-hairy images from the ISIC 2019 dataset.
- Removed hair artifacts using the `src/hair_remove.py` computer-vision pipeline and merged results with clean images.
- Enhanced images using `src/image_enhance.py` to improve contrast and consistency where needed.
- Resized all images to a standard **720×720** resolution and organized them into clear folders (`final_dataset/BCC`, `final_dataset/BKL`, `final_dataset/MEL`, `final_dataset/NV`).
- Applied data augmentation (rotation, flips, color jitter, etc.) to reduce class imbalance and increase training robustness.
- Created separate cleaned and hairy data folders under `Data_Prep/` for transparency and reproducibility.

## Why this matters 💡
- Hair artifacts can bias or reduce performance of lesion classifiers—removing them reduces noise and improves feature extraction.
- Standardizing image size simplifies model architectures and training pipelines.
- Addressing class imbalance via augmentation helps reduce overfitting to majority classes and improves generalization to under-represented lesion types.
- Clear folder structure (`Cleaned_720d_Dermo_Img`, `Data_Prep`, `final_dataset`) improves reproducibility and makes it easy to extend the pipeline.

## How to reproduce 🔧
1. Ensure you have the ISIC 2019 images available in the expected raw location (set paths in the scripts if different).
2. Run `python src/hair_remove.py` to produce hair-removed outputs (check script args for input/output paths).
3. Optionally run `python src/image_enhance.py` to apply contrast/normalization steps.
4. Use the augmentation and resizing utilities in `Data_Prep/` (or the Jupyter notebook `src/notebook.ipynb`) to create the final 720×720 dataset in `final_dataset/`.

> Note: Check script headers and notebook cells for required dependencies and environment details.

## Next steps ▶️
- Validate the cleaned dataset by sampling images from each class and confirming hair removal/quality.
- Train baseline classification models (e.g., ResNet, EfficientNet) to measure improvement vs raw images.
- Add automated tests to verify output folder structure, image sizes, and a small visual check for hair artifacts.
- Experiment with additional augmentation strategies and class rebalancing techniques.

## Files of interest
- `src/hair_remove.py` – hair detection and removal pipeline
- `src/image_enhance.py` – image enhancement utilities
- `src/notebook.ipynb` – exploratory notebook and reproducible 



