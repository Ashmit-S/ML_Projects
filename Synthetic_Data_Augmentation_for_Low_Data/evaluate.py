import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_classification(rows):
    if len(rows) == 0:
        print("No classification results found.")
        return

    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS (Accuracy and F1)")
    print("=" * 80)

    grouped = defaultdict(list)
    for r in rows:
        key = (r["model"], int(r["num_real"]))
        grouped[key].append(r)

    low_data_positive = 0
    high_data_non_positive = 0

    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        model_name, n_real = key
        group = grouped[key]
        real = None
        mixed = []
        for g in group:
            if int(g["synth_count"]) == 0:
                real = g
            else:
                mixed.append(g)

        print(f"\nModel={model_name} | real_samples={n_real}")
        if real is None:
            print("  Missing real-only baseline")
            continue

        real_f1 = float(real["f1"])
        real_acc = float(real["accuracy"])
        print(f"  real-only: acc={real_acc:.4f}, f1={real_f1:.4f}")

        best_mixed = None
        for m in mixed:
            if best_mixed is None or float(m["f1"]) > float(best_mixed["f1"]):
                best_mixed = m
            print(
                f"  mixed +{m['synth_count']} synth: acc={float(m['accuracy']):.4f}, f1={float(m['f1']):.4f}"
            )

        if best_mixed is not None:
            delta = float(best_mixed["f1"]) - real_f1
            print(f"  best mixed delta F1: {delta:+.4f}")
            if n_real <= 40 and delta > 0:
                low_data_positive += 1
            if n_real >= 60 and delta <= 0:
                high_data_non_positive += 1

    print("\nTrend notes (classification):")
    print(f"  low-data (<=40) with positive gain count: {low_data_positive}")
    print(f"  higher-data (>=60) with plateau/drop count: {high_data_non_positive}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_csv", type=str, default="results/classification_results.csv")
    args = parser.parse_args()

    cls_rows = read_csv_rows(args.cls_csv)
    summarize_classification(cls_rows)


if __name__ == "__main__":
    main()
