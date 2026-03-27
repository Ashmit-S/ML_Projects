from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune YOLOv8 on PPE dataset and evaluate on held-out validation split "
            "with mAP@0.5, precision, and recall."
        )
    )
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="")
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--target-classes",
        default="person,hard_hat,helmet,vest,safety_vest",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--save-json", default="")
    parser.add_argument(
        "--weights",
        default="",
        help="Path to trained weights (.pt). If provided, training is skipped.",
    )
    return parser.parse_args()


def load_dataset_names(data_yaml_path: Path) -> Dict[int, str]:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names_raw = data.get("names", [])
    if isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    elif isinstance(names_raw, list):
        names = {i: str(v) for i, v in enumerate(names_raw)}
    else:
        raise ValueError("Invalid names format in data.yaml")

    if not names:
        raise ValueError("No class names found in data.yaml")
    return names


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_target_class_ids(
    dataset_names: Dict[int, str], target_class_names: Sequence[str]
) -> Tuple[List[int], List[str]]:
    wanted = {normalize_name(x) for x in target_class_names if x.strip()}

    resolved: List[int] = []
    missing: List[str] = []

    normalized_to_id = {
        normalize_name(class_name): class_id for class_id, class_name in dataset_names.items()
    }

    for name in sorted(wanted):
        class_id = normalized_to_id.get(name)
        if class_id is None:
            missing.append(name)
        else:
            resolved.append(class_id)

    return sorted(set(resolved)), missing


def ensure_weights_path(project: Path, name: str) -> Path:
    best = project / name / "weights" / "best.pt"
    last = project / name / "weights" / "last.pt"

    if best.exists():
        return best
    if last.exists():
        return last

    raise FileNotFoundError(
        f"No trained weights found in expected path: {best} or {last}."
    )


def extract_core_metrics(metrics) -> Dict[str, float]:
    return {
        "map50": float(getattr(metrics.box, "map50", 0.0)),
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
    }


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    target_names = [x.strip() for x in args.target_classes.split(",") if x.strip()]
    dataset_names = load_dataset_names(data_path)
    class_ids, missing_names = resolve_target_class_ids(dataset_names, target_names)

    if not class_ids:
        raise ValueError("None of the requested target classes exist in data.yaml.")

    selected_name_map = {class_id: dataset_names[class_id] for class_id in class_ids}

    print(f"Data config: {data_path}")
    print(f"Selected class IDs: {class_ids}")
    print(f"Selected classes: {list(selected_name_map.values())}")
    if missing_names:
        print(f"Missing requested classes (ignored): {missing_names}")

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Provided weights not found: {weights_path}")
        print(f"Skipping training. Using existing weights: {weights_path}")
    else:
        print("Starting fine-tuning run")
        print(f"Base model: {args.model}")
        model = YOLO(args.model)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            classes=class_ids,
        )
        weights_path = ensure_weights_path(Path(args.project), args.name)
        print(f"Using trained weights for validation: {weights_path}")

    tuned_model = YOLO(str(weights_path))
    metrics = tuned_model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        classes=class_ids,
        verbose=False,
    )

    score = extract_core_metrics(metrics)

    print("\nValidation Metrics")
    print(f"mAP@0.5 : {score['map50']:.4f}")
    print(f"Precision: {score['precision']:.4f}")
    print(f"Recall   : {score['recall']:.4f}")

    summary = {
        "data": str(data_path),
        "trained_weights": str(weights_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "selected_class_ids": class_ids,
        "selected_class_names": [selected_name_map[i] for i in class_ids],
        "ignored_missing_target_classes": missing_names,
        "metrics": score,
    }

    if args.save_json:
        output_json = Path(args.save_json)
    else:
        output_json = weights_path.parent.parent / f"metrics_{args.split}.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metrics summary: {output_json}")


if __name__ == "__main__":
    main()