"""Step 5 — hyperparameter tuning with cross-validation.

Tunes the XGBoost pipeline with a randomized search over a stratified
cross-validation on the TRAINING set only (the test set is never touched during
tuning, so there is no leakage). We optimize AUPRC, which is more informative
than accuracy for an imbalanced outcome. The whole preprocessing + model is
wrapped in a single Pipeline so each CV fold re-fits the preprocessing correctly.

Reports the cross-validated score (mean +/- std) and the held-out test
performance, and saves the best parameters.

Usage:
    python -m src.models.tune --config configs/model.yaml --tune-config configs/tune.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_metrics
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


def build_search_space(tune_cfg: dict, scale_pos_weight: float) -> dict:
    """Prefix each hyperparameter with 'clf__' to target the model in the Pipeline."""
    space = {f"clf__{k}": v for k, v in tune_cfg["search_space"].items()}
    # Let the search decide whether class-imbalance weighting helps.
    space["clf__scale_pos_weight"] = [1.0, scale_pos_weight]
    return space


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the XGBoost model with CV.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--tune-config", default="configs/tune.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    tcfg = load_yaml(args.tune_config)
    seed = cfg["split"]["random_state"]

    # Data prep (identical to training).
    df = pd.read_parquet(cfg["input"]["processed_file"])
    df = apply_filters(df, cfg)
    df = dedup_first_encounter(df, cfg)
    x_raw, y, groups = split_columns(df, cfg)
    x_tr, x_te, y_tr, y_te, _g_tr, _g_te = train_test_split_patient(x_raw, y, groups, df, cfg)

    pos = int(y_tr.sum())
    spw = (len(y_tr) - pos) / pos if pos else 1.0

    pipe = Pipeline([
        ("pre", build_preprocessor(x_tr, cfg)),
        ("clf", XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipe,
        build_search_space(tcfg, spw),
        n_iter=tcfg["n_iter"],
        scoring=tcfg["scoring"],
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )

    print(f"Running randomized search ({tcfg['n_iter']} candidates x {tcfg['cv_folds']} folds) ...")
    search.fit(x_tr, y_tr)

    best_mean = search.best_score_
    best_std = search.cv_results_["std_test_score"][search.best_index_]
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

    # Held-out test performance of the tuned model.
    p_test = search.predict_proba(x_te)[:, 1]
    test_metrics = compute_metrics(y_te, p_test, cfg["evaluation"]["threshold"])

    print(f"\nBest CV {tcfg['scoring']}: {best_mean:.4f} +/- {best_std:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("\nHeld-out test metrics (tuned model):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Persist results.
    Path(tcfg["paths"]["best_params_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(tcfg["paths"]["best_params_file"], "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    with open(tcfg["paths"]["tuning_metrics_file"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_scoring": tcfg["scoring"],
                "cv_mean": round(float(best_mean), 4),
                "cv_std": round(float(best_std), 4),
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSaved best params to {tcfg['paths']['best_params_file']}")


if __name__ == "__main__":
    main()
