import csv
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def copy_many(files, out_dir: Path):
    ensure_dir(out_dir)
    for f in files:
        shutil.copy2(f, out_dir / f.name)


def copy_with_labels(image_files, image_out: Path, label_out: Path, label_suffix: str = ".txt"):
    ensure_dir(image_out)
    ensure_dir(label_out)
    for img_path in image_files:
        shutil.copy2(img_path, image_out / img_path.name)
        label_name = img_path.stem + label_suffix
        src_label = img_path.parent.parent / "labels" / label_name
        if src_label.exists():
            shutil.copy2(src_label, label_out / label_name)


def write_rows_csv(path: Path, rows, fieldnames):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
