import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import models

from dataset import build_classification_subset
from utils import set_seed, write_rows_csv


def get_model(model_name: str, num_classes: int):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model_name: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def predict_labels(model, loader, device):
    model.eval()
    ys_true = []
    ys_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            ys_pred.extend(preds)
            ys_true.extend(labels.tolist())
    return ys_true, ys_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_root", type=str, default="data/classification/real")
    parser.add_argument("--synthetic_root", type=str, default="data/synthetic/classification/train")
    parser.add_argument("--work_dir", type=str, default="data/exp_cache/classification_subsets")
    parser.add_argument("--results_csv", type=str, default="results/classification_results.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    real_sizes = [20, 50, 100, 200]
    synth_counts = [0, 20, 40, 60, 80]
    model_names = ["resnet18", "mobilenet_v2"]

    rows = []

    for model_name in model_names:
        for num_real in real_sizes:
            for synth_count in synth_counts:
                setup_name = "real-only" if synth_count == 0 else f"real+synthetic_{synth_count}"
                print("=" * 70)
                print(f"Model={model_name} | num_real={num_real} | setup={setup_name}")

                train_ds, val_ds, test_ds, subset_dir = build_classification_subset(
                    real_root=args.real_root,
                    synthetic_root=args.synthetic_root,
                    out_root=args.work_dir,
                    num_real=num_real,
                    synth_count=synth_count,
                    seed=args.seed,
                )
                print(f"Subset folder: {subset_dir}")
                print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Test size: {len(test_ds)}")

                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

                model = get_model(model_name, num_classes=2).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                for epoch in range(args.epochs):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    y_val_true, y_val_pred = predict_labels(model, val_loader, device)
                    val_acc = accuracy_score(y_val_true, y_val_pred) if len(y_val_true) > 0 else 0.0
                    print(f"Epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.4f} - val_acc={val_acc:.4f}")

                y_true, y_pred = predict_labels(model, test_loader, device)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="binary")

                row = {
                    "model": model_name,
                    "num_real": num_real,
                    "synth_count": synth_count,
                    "setup": setup_name,
                    "accuracy": round(float(acc), 4),
                    "f1": round(float(f1), 4),
                    "train_size": len(train_ds),
                }
                rows.append(row)
                print(f"Test Accuracy={acc:.4f} | F1={f1:.4f}")

    out_path = Path(args.results_csv)
    write_rows_csv(
        out_path,
        rows,
        fieldnames=["model", "num_real", "synth_count", "setup", "accuracy", "f1", "train_size"],
    )

    print("\nSaved classification results to:", out_path)
    print("Quick view:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
