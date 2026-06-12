#!/usr/bin/env python3
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Minimal external trainer for case8")
    parser.add_argument("--config", required=True, help="Path to input hyperparameter config JSON")
    parser.add_argument("--output", required=True, help="Path to output result JSON")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lr = float(cfg.get("learning_rate", 0.01))
    num_trees = int(cfg.get("num_trees", 100))
    max_depth = int(cfg.get("max_depth", 6))
    subsample = float(cfg.get("subsample", 0.8))

    score = 0.55
    score += min(num_trees / 1000.0, 0.25)
    score += min(max_depth / 50.0, 0.15)
    score += max(0.0, min(subsample, 1.0)) * 0.05
    score += max(0.0, 0.02 - abs(lr - 0.02))
    score = max(0.01, min(score, 0.99))

    train_loss_history = [float(max(1.0 - score + 0.12, 0.0)), float(max(1.0 - score + 0.06, 0.0)), float(max(1.0 - score + 0.02, 0.0))]
    val_loss_history = [float(max(1.0 - score + 0.15, 0.0)), float(max(1.0 - score + 0.08, 0.0)), float(max(1.0 - score, 0.0))]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "external_model.bin")
    with open(model_path, "wb") as mf:
        mf.write(b"dummy external model")

    result = {
        "val_accuracy": float(score),
        "model_path": model_path,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
