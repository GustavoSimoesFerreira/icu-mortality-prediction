"""Operating-threshold analysis for the ICU mortality model.

The model outputs probabilities; turning them into a decision needs a threshold.
The default 0.5 is arbitrary. For ICU mortality, missing a death (false negative)
is worse than a false alarm, so we prefer a threshold that reaches a high recall.

This script sweeps thresholds, reports the recall / precision / specificity
trade-off, saves a plot, and recommends the threshold that meets a TARGET RECALL
with the best precision available at that recall.

Usage:
    python -m src.evaluation.threshold_analysis \\
        --config configs/model.yaml --threshold-config configs/threshold.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from src.features.preprocess import (
    apply_filters,
    build_preprocessor,
    dedup_first_encounter,
    split_columns,
    train_test_split_patient,
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sweep_thresholds(y_true, y_prob, grid) -> list[dict]:
    """Compute recall/precision/specificity/F1 at each threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": round(float(t), 3),
                "recall": round(float(recall), 4),
                "precision": round(float(precision), 4),
                "specificity": round(float(specificity), 4),
                "f1": round(float(f1), 4),
            }
        )
    return rows


def recommend_threshold(rows: list[dict], target_recall: float) -> dict | None:
    """Highest threshold whose recall still meets the target (best precision)."""
    meeting = [r for r in rows if r["recall"] >= target_recall]
    return max(meeting, key=lambda r: r["threshold"]) if meeting else None


def get_test_predictions(cfg: dict):
    """Train the XGBoost model (as in train.py) and return test probabilities."""
    seed = cfg["split"]["random_state"]
    df = pd.read_parquet(cfg["input"]["processed_file"])
    df = apply_filters(df, cfg)
    df = dedup_first_encounter(df, cfg)
    x_raw, y, groups = split_columns(df, cfg)
    x_tr, x_te, y_tr, y_te, _g_tr, _g_te = train_test_split_patient(x_raw, y, groups, df, cfg)

    pre = build_preprocessor(x_tr, cfg)
    xt_tr = pre.fit_transform(x_tr)
    xt_te = pre.transform(x_te)
    pos = int(y_tr.sum())
    spw = (len(y_tr) - pos) / pos if pos else 1.0
    model = XGBClassifier(
        **cfg["model"]["xgboost"],
        scale_pos_weight=spw,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
    )
    model.fit(xt_tr, y_tr)
    return y_te, model.predict_proba(xt_te)[:, 1]


def save_plot(rows: list[dict], recommended: dict | None, out_path: Path) -> None:
    thresholds = [r["threshold"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, [r["recall"] for r in rows], label="Recall (sensitivity)")
    plt.plot(thresholds, [r["precision"] for r in rows], label="Precision")
    plt.plot(thresholds, [r["specificity"] for r in rows], label="Specificity")
    if recommended:
        plt.axvline(
            recommended["threshold"], color="k", linestyle="--",
            label=f"Recommended = {recommended['threshold']}",
        )
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric")
    plt.title("Operating-threshold trade-off")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose an operating threshold.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--threshold-config", default="configs/threshold.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    tcfg = load_yaml(args.threshold_config)

    y_te, p_te = get_test_predictions(cfg)
    g = tcfg["grid"]
    grid = np.arange(g["start"], g["stop"] + 1e-9, g["step"])
    rows = sweep_thresholds(y_te, p_te, grid)
    target = tcfg["target_recall"]
    rec = recommend_threshold(rows, target)

    # Print a compact table at 0.05 steps.
    print(f"\nTarget recall: {target}\n")
    print(f"{'thr':>6} {'recall':>8} {'precision':>10} {'specificity':>12} {'f1':>7}")
    for r in rows:
        if abs((r["threshold"] / 0.05) - round(r["threshold"] / 0.05)) < 1e-6:
            print(f"{r['threshold']:>6} {r['recall']:>8} {r['precision']:>10} "
                  f"{r['specificity']:>12} {r['f1']:>7}")

    if rec:
        print(f"\nRecommended threshold: {rec['threshold']} "
              f"-> recall {rec['recall']}, precision {rec['precision']}, "
              f"specificity {rec['specificity']}")
        print(f"(vs default 0.5, this captures {rec['recall']:.0%} of ICU deaths)")
    else:
        print(f"\nNo threshold reaches recall {target}; consider a lower target.")

    save_plot(rows, rec, Path(tcfg["paths"]["figure_file"]))
    Path(tcfg["paths"]["results_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(tcfg["paths"]["results_file"], "w", encoding="utf-8") as f:
        json.dump({"target_recall": target, "recommended": rec, "sweep": rows}, f, indent=2)
    print(f"\nSaved plot to {tcfg['paths']['figure_file']}")


if __name__ == "__main__":
    main()
