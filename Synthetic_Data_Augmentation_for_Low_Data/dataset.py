import random
from pathlib import Path

from torchvision import datasets, transforms

from utils import copy_many, copy_with_labels, ensure_dir, list_images, reset_dir, set_seed


def _sample_balanced_classification(train_dir: Path, num_real: int, seed: int):
    class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs)
    n_classes = len(class_dirs)
    if n_classes == 0:
        raise ValueError(f"No class folders found in {train_dir}")

    base = num_real // n_classes
    remainder = num_real % n_classes

    sampled = {}
    rng = random.Random(seed)
    for i, cdir in enumerate(class_dirs):
        k = base + (1 if i < remainder else 0)
        images = list_images(cdir)
        if k > len(images):
            k = len(images)
        sampled[cdir.name] = rng.sample(images, k=k)
    return sampled


def build_classification_subset(
    real_root,
    synthetic_root,
    out_root,
    num_real=20,
    synth_count=0,
    seed=42,
):
    set_seed(seed)
    real_root = Path(real_root)
    out_root = Path(out_root)

    train_real = real_root / "train"
    val_real = real_root / "val"
    test_real = real_root / "test"

    run_dir = out_root / f"n{num_real}_synth{synth_count}_seed{seed}"
    train_out = run_dir / "train"
    val_out = run_dir / "val"
    test_out = run_dir / "test"

    reset_dir(run_dir)

    sampled = _sample_balanced_classification(train_real, num_real=num_real, seed=seed)

    for class_name, files in sampled.items():
        copy_many(files, train_out / class_name)

    # Val/test are always fully real to measure generalization.
    for split_src, split_out in [(val_real, val_out), (test_real, test_out)]:
        class_dirs = [p for p in split_src.iterdir() if p.is_dir()]
        for cdir in class_dirs:
            copy_many(list_images(cdir), split_out / cdir.name)

    if synth_count > 0:
        if synthetic_root is None:
            print("synthetic_root is None, skipping synthetic add")
        else:
            synthetic_root = Path(synthetic_root)
            rng = random.Random(seed + 7)
            n_classes = max(len(sampled), 1)
            per_class = synth_count // n_classes
            remainder = synth_count % n_classes

            for i, class_name in enumerate(sampled.keys()):
                synth_dir = synthetic_root / class_name
                if not synth_dir.exists():
                    print(f"No synthetic folder for class {class_name}, skipping")
                    continue
                synth_imgs = list_images(synth_dir)
                if len(synth_imgs) == 0:
                    continue
                target_k = per_class + (1 if i < remainder else 0)
                k = min(target_k, len(synth_imgs))
                chosen = rng.sample(synth_imgs, k=k)
                copy_many(chosen, train_out / class_name)

    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(train_out, transform=transform_train)
    val_ds = datasets.ImageFolder(val_out, transform=transform_eval)
    test_ds = datasets.ImageFolder(test_out, transform=transform_eval)

    return train_ds, val_ds, test_ds, run_dir


def build_detection_subset(
    real_yolo_root,
    out_root,
    num_real=20,
    synth_ratio=0,
    synthetic_yolo_root=None,
    seed=42,
):
    set_seed(seed)
    real_yolo_root = Path(real_yolo_root)
    out_root = Path(out_root)
    run_dir = out_root / f"n{num_real}_ratio{synth_ratio}_seed{seed}"
    reset_dir(run_dir)

    train_img_src = real_yolo_root / "train" / "images"
    val_img_src = real_yolo_root / "valid" / "images"
    test_img_src = real_yolo_root / "test" / "images"

    train_img_out = run_dir / "train" / "images"
    train_lbl_out = run_dir / "train" / "labels"

    val_img_out = run_dir / "valid" / "images"
    val_lbl_out = run_dir / "valid" / "labels"

    test_img_out = run_dir / "test" / "images"
    test_lbl_out = run_dir / "test" / "labels"

    train_images = list_images(train_img_src)
    rng = random.Random(seed)
    k_real = min(num_real, len(train_images))
    chosen_real = rng.sample(train_images, k=k_real)
    copy_with_labels(chosen_real, train_img_out, train_lbl_out)

    # Keep val/test fully real.
    copy_with_labels(list_images(val_img_src), val_img_out, val_lbl_out)
    copy_with_labels(list_images(test_img_src), test_img_out, test_lbl_out)

    if synth_ratio > 0 and synthetic_yolo_root is not None:
        synthetic_yolo_root = Path(synthetic_yolo_root)
        synth_img_src = synthetic_yolo_root / "train" / "images"
        if synth_img_src.exists():
            synth_images = list_images(synth_img_src)
            k_synth = min(int(k_real * synth_ratio), len(synth_images))
            chosen_synth = rng.sample(synth_images, k=k_synth)
            copy_with_labels(chosen_synth, train_img_out, train_lbl_out)
        else:
            print(f"Synthetic detection images not found at {synth_img_src}, using real-only")

    data_yaml = run_dir / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {run_dir.as_posix()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            "nc: 1",
            "names: ['helmet']",
        ]
    )
    ensure_dir(data_yaml.parent)
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return data_yaml, run_dir
