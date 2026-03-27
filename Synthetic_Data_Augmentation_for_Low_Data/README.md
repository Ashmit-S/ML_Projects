# Synthetic Data Augmentation in Low-Data Vision

## Problem motivation (low-data learning)
When we only have a small number of real images, vision models can overfit and fail to generalize.
This mini project studies if synthetic images from Stable Diffusion can help in low-data settings.

## Approach (synthetic augmentation)
1. Train with very small real subsets: 20, 40, and 60 samples.
2. Generate synthetic images using Stable Diffusion v1.5 with prompts like:
   - "a person wearing a safety helmet"
   - "a person without a helmet"
3. Compare real-only training vs real + synthetic training with fixed synthetic image counts (20, 40, 60, 80).
4. Keep validation and test fully real to measure real-world generalization.

## Models used
- Classification:
  - ResNet-18
  - MobileNetV2

## Project files
- `generate.py` - generate synthetic images using Stable Diffusion v1.5
- `dataset.py` - build low-data subsets and combine real + synthetic data
- `train_classification.py` - train ResNet-18 and MobileNetV2
- `evaluate.py` - print comparison tables and trend summary
- `utils.py` - helper functions

## Suggested folder setup

### Real classification data
```
data/classification/real/
  train/
    helmet/
    no_helmet/
  val/
    helmet/
    no_helmet/
  test/
    helmet/
    no_helmet/
```

### Synthetic classification data (generated)
```
data/synthetic/classification/train/
  helmet/
  no_helmet/
```

## Install
```bash
pip install -r requirements.txt
```

## Run experiments

### 1) Generate synthetic images
```bash
python generate.py --num_per_class 80
```

### 2) Train classification models
```bash
python train_classification.py --epochs 3 --batch_size 16
```

### 3) Print comparison tables and trends
```bash
python evaluate.py
```

## Evaluation setup (real vs mixed)
- Classification metrics:
  - Accuracy
  - F1-score

The scripts save CSVs in `results/` and then `evaluate.py` summarizes:
- real-only vs mixed
- trend in low-data vs higher-data settings

## Key insight
This project is intended to show behavior trends, not SOTA numbers.
Expected behavior:
- Synthetic data can help when real data is very limited.
- Gains may plateau (or even drop) as real data grows.
- Distribution mismatch and visual artifacts can hurt performance.

## Notes about limitations
- Stable Diffusion images can have unrealistic textures, backgrounds, or shapes.
- Prompt bias can shift image distribution away from real test data.

This is why we evaluate only on fully real validation/test sets.
