import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    print("=" * 80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()

    # Global controls
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip_convert", action="store_true")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_classification", action="store_true")
    parser.add_argument("--skip_evaluate", action="store_true")

    # Shared dataset paths
    parser.add_argument("--yolo_root", type=str, default="Helmet Detection.v5i.yolov8")
    parser.add_argument("--cls_real_root", type=str, default="data/classification/real")
    parser.add_argument("--clear_cls_output", action="store_true")

    # Generate settings
    parser.add_argument("--num_per_class", type=int, default=80)
    parser.add_argument("--gen_steps", type=int, default=30)
    parser.add_argument("--gen_seed", type=int, default=42)

    # Classification settings
    parser.add_argument("--cls_epochs", type=int, default=3)
    parser.add_argument("--cls_batch_size", type=int, default=16)
    parser.add_argument("--cls_lr", type=float, default=1e-3)
    parser.add_argument("--cls_seed", type=int, default=42)

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    print("Project root:", root)

    if not args.skip_convert:
        cmd = [
            args.python,
            str(root / "convert_yolo_to_classification.py"),
            "--yolo_root",
            args.yolo_root,
            "--output_root",
            args.cls_real_root,
        ]
        if args.clear_cls_output:
            cmd.append("--clear_output")
        run_cmd(cmd)
    else:
        print("Skipping YOLO->classification conversion")

    if not args.skip_generate:
        run_cmd(
            [
                args.python,
                str(root / "generate.py"),
                "--num_per_class",
                str(args.num_per_class),
                "--steps",
                str(args.gen_steps),
                "--seed",
                str(args.gen_seed),
            ]
        )
    else:
        print("Skipping synthetic generation")

    if not args.skip_classification:
        run_cmd(
            [
                args.python,
                str(root / "train_classification.py"),
                "--real_root",
                args.cls_real_root,
                "--epochs",
                str(args.cls_epochs),
                "--batch_size",
                str(args.cls_batch_size),
                "--lr",
                str(args.cls_lr),
                "--seed",
                str(args.cls_seed),
            ]
        )
    else:
        print("Skipping classification")

    if not args.skip_evaluate:
        run_cmd([args.python, str(root / "evaluate.py")])
    else:
        print("Skipping evaluation")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
