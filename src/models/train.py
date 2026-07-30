"""
Step 2 — train and evaluate the first predictive model.

Pipeline:
  1. load the processed data,
  2. filter + reduce to one row per patient,
  3. split at the patient level,
  4. train a logistic-regression baseline and an XGBoost model,
  5. evaluate (AUROC, AUPRC, Brier, calibration),
  6. audit fairness across subgroups,
  7. explain the model with SHAP,
  8. save metrics + figures under reports/.

Usage:
    python -m src.models.train --config configs/model.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_metrics, fairness_by_group
from src.explain.shap_summary import save_shap_summary
from src.features.preprocess import (
    apply_filters,
    build_preprocessor,
    dedup_first_encounter,
    split_columns,
    train_test_split_patient,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scale_pos_weight(y) -> float:
    """Ratio of negatives to positives — used to tell XGBoost about imbalance."""
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return neg / pos if pos else 1.0


def save_calibration_plot(curves: dict, out_path: Path) -> None:
    """Plot calibration curves for the trained models."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name, (prob_true, prob_pred) in curves.items():
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Step 2 model.")
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 1-2. Load + prepare.
    df = pd.read_parquet(cfg["input"]["processed_file"])
    df = apply_filters(df, cfg)
    df = dedup_first_encounter(df, cfg)
    x_raw, y, groups = split_columns(df, cfg)

    # 3. Patient-level split (also splits the readable group columns).
    x_tr, x_te, y_tr, y_te, _g_tr, g_te = train_test_split_patient(
        x_raw, y, groups, df, cfg
    )
    print(f"Train rows: {len(x_tr):,} | Test rows: {len(x_te):,} "
          f"| Positive rate: {y.mean():.1%}")

    # 4. Fit preprocessing on train only, then transform both.
    pre = build_preprocessor(x_raw, cfg)
    xt_tr = pre.fit_transform(x_tr)
    xt_te = pre.transform(x_te)
    feature_names = pre.get_feature_names_out()

    threshold = cfg["evaluation"]["threshold"]
    results, calibration = {}, {}

    # 4a. Logistic-regression baseline.
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(xt_tr, y_tr)
    p_lr = logreg.predict_proba(xt_te)[:, 1]
    results["logistic_regression"] = compute_metrics(y_te, p_lr, threshold)
    calibration["Logistic Regression"] = calibration_curve(y_te, p_lr, n_bins=10)

    # 4b. XGBoost.
    xgb_cfg = cfg["model"]["xgboost"]
    xgb = XGBClassifier(
        **xgb_cfg,
        scale_pos_weight=scale_pos_weight(y_tr),
        eval_metric="logloss",
        tree_method="hist",
        random_state=cfg["split"]["random_state"],
    )
    xgb.fit(xt_tr, y_tr)
    p_xgb = xgb.predict_proba(xt_te)[:, 1]
    results["xgboost"] = compute_metrics(y_te, p_xgb, threshold)
    calibration["XGBoost"] = calibration_curve(y_te, p_xgb, n_bins=10)

    # 5. Report headline metrics.
    metrics_df = pd.DataFrame(results).T.reset_index(names="model")
    print_table("MODEL COMPARISON", metrics_df)

    # 6. Fairness audit on the stronger model (XGBoost).
    fairness_df = fairness_by_group(
        y_te, p_xgb, g_te, cfg["fairness"]["group_columns"]
    )
    print_table("FAIRNESS AUDIT (XGBoost, AUROC per subgroup)", fairness_df)

    # 7. SHAP explanation for XGBoost.
    figures_dir = Path(cfg["paths"]["figures_dir"])
    shap_path = save_shap_summary(
        xgb, xt_te, feature_names,
        figures_dir / "shap_summary.png",
        sample_size=cfg["explain"]["sample_size"],
    )
    save_calibration_plot(calibration, figures_dir / "calibration.png")

    # 8. Persist metrics.
    out = {
        "metrics": results,
        "fairness": fairness_df.to_dict(orient="records"),
        "n_train": len(x_tr),
        "n_test": len(x_te),
    }
    metrics_file = Path(cfg["paths"]["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved metrics to {metrics_file}")
    print(f"Saved figures to {figures_dir}/ (shap_summary.png, calibration.png)")
    print(f"SHAP summary: {shap_path}")


if __name__ == "__main__":
    main()
