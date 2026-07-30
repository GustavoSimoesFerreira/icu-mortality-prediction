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


DATA_URL = "https://physionet.org/files/mimic2-iaccd/1.0/full_cohort_data.csv"

def download_dataset(csv_path: str) -> pd.DataFrame:
    """Load the Open Access PhysioNet IAC dataset.

    Reads from a local cache if present; otherwise downloads it once from
    PhysioNet's public URL and caches it. No manual download, no credentialing.
    """
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)

    print(f"Downloading dataset from {DATA_URL} ...")
    df = pd.read_csv(DATA_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Ensure the target is 0/1 and derive an age band for the fairness audit.

    This CSV encodes missing values as blank/NA (read as NaN by pandas), so
    there is no '?' token to replace.
    """
    target = cfg["target"]["column"]
    df[target] = df[target].astype(int)

    # age_band is used only for the fairness audit (age itself stays a feature).
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 70, 85, 200],
        labels=["<40", "40-54", "55-69", "70-84", "85+"],
    )
    return df


def summarize(df: pd.DataFrame, cfg: dict) -> None:
    """Print a quick exploratory summary — this is the point of Step 1."""
    target = cfg["target"]["column"]

    print("\n" + "=" * 60)
    print(f"{cfg['dataset']['name'].upper()} — EXPLORATORY SUMMARY")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")

    counts = df[target].value_counts().sort_index()
    print("\nTarget distribution (28-day mortality):")
    for value, n in counts.items():
        label = "died <=28d" if value == 1 else "survived"
        print(f"  {value} = {label:<14} {n:>7,}  ({n / len(df):.1%})")

    # Missingness
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

    print(f"Loading dataset '{cfg['dataset']['name']}' ...")
    df = download_dataset(cfg["dataset"]["raw_file"])
    df = clean(df, cfg)
    summarize(df, cfg)

    # Save the processed copy (folder is gitignored — data is never committed).
    out_path = Path(cfg["paths"]["processed_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved processed data to: {out_path}")


if __name__ == "__main__":
    main()
