"""
Step 1 — Load and explore the UCI Diabetes 130-US Hospitals dataset.

This script downloads the dataset, does light cleaning, builds a binary
readmission target, prints a quick exploratory summary, and saves a processed
copy. No modeling happens here — the goal is a working `data -> processed file`
flow that we can build on.

Usage:
    python -m src.data.load_diabetes --config configs/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Read the YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_binary_target(series: pd.Series, positive_value: str) -> pd.Series:
    """Turn the 'readmitted' column into a 0/1 target.

    1 = readmitted within 30 days (positive_value), 0 = otherwise.
    Kept as a small pure function so it can be unit-tested without any network.
    """
    return (series == positive_value).astype(int)


def download_dataset(uci_id: int) -> pd.DataFrame:
    """Fetch the dataset from the UCI ML Repository and return one DataFrame.

    The ucimlrepo package splits the data into ID / feature / target frames;
    we stitch them back together into a single tidy DataFrame.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'ucimlrepo' package is missing. Run: pip install -r requirements.txt"
        ) from exc

    try:
        dataset = fetch_ucirepo(id=uci_id)
    except Exception as exc:  # network / UCI availability issues
        raise SystemExit(
            f"Could not download dataset id={uci_id} from the UCI repository.\n"
            f"Check your internet connection and try again.\nOriginal error: {exc}"
        ) from exc

    frames = [f for f in (dataset.data.ids, dataset.data.features, dataset.data.targets)
              if f is not None]
    df = pd.concat(frames, axis=1)
    return df


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Light cleaning: mark missing values and drop unusable columns."""
    missing_token = cfg["cleaning"]["missing_token"]
    df = df.replace(missing_token, pd.NA)

    drop_cols = [c for c in cfg["cleaning"].get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Build the binary target.
    src = cfg["target"]["source_column"]
    df[cfg["target"]["binary_column"]] = make_binary_target(
        df[src], cfg["target"]["positive_value"]
    )
    return df


def summarize(df: pd.DataFrame, cfg: dict) -> None:
    """Print a quick exploratory summary — this is the point of Step 1."""
    target = cfg["target"]["binary_column"]

    print("\n" + "=" * 60)
    print("DIABETES 130-US — EXPLORATORY SUMMARY")
    print("=" * 60)
    print(f"Rows (hospital encounters): {len(df):,}")
    print(f"Columns: {df.shape[1]}")

    # Class balance — medical outcomes are usually imbalanced.
    counts = df[target].value_counts().sort_index()
    print("\nTarget distribution (readmitted within 30 days):")
    for value, n in counts.items():
        label = "readmitted <30d" if value == 1 else "not (<30d)"
        print(f"  {value} = {label:<18} {n:>8,}  ({n / len(df):.1%})")

    # Leakage warning: the same patient appears in multiple encounters.
    if "patient_nbr" in df.columns:
        n_patients = df["patient_nbr"].nunique()
        print(f"\nUnique patients: {n_patients:,}  (vs {len(df):,} encounters)")
        print("  --> For Step 2, split at the PATIENT level to avoid leakage.")

    # Missingness — helps decide what to impute or drop later.
    miss = (df.isna().mean().sort_values(ascending=False) * 100).round(1)
    top_missing = miss[miss > 0].head(8)
    if len(top_missing):
        print("\nTop columns by % missing:")
        for col, pct in top_missing.items():
            print(f"  {col:<24} {pct:>5}%")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load & explore the diabetes dataset.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Downloading dataset id={cfg['dataset']['uci_id']} from UCI...")
    df = download_dataset(cfg["dataset"]["uci_id"])
    df = clean(df, cfg)
    summarize(df, cfg)

    # Save the processed copy (folder is gitignored — data is never committed).
    out_path = Path(cfg["paths"]["processed_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved processed data to: {out_path}")


if __name__ == "__main__":
    main()
