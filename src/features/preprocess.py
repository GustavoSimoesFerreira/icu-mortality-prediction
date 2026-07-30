"""
Feature preprocessing for Step 2.

Responsibilities:
  * filter out encounters that shouldn't be modeled (e.g. expired/hospice),
  * reduce to one row per patient (avoids leakage + within-patient correlation),
  * split into train/test at the PATIENT level,
  * build an sklearn transformer that encodes categoricals and scales numerics.

Keeping these as small functions makes each one unit-testable without network.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop encounters listed in the filter config (e.g. expired / hospice)."""
    col = cfg["filters"]["discharge_column"]
    exclude = cfg["filters"]["exclude_discharge_ids"]
    if col in df.columns:
        df = df[~df[col].isin(exclude)].copy()
    return df


def dedup_first_encounter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep the first encounter per patient, if enabled in config."""
    if not cfg["patient"].get("dedup_first_encounter", False):
        return df
    id_col = cfg["patient"]["id_column"]
    if id_col not in df.columns:
        return df
    # 'encounter_id' is roughly chronological, so sorting by it approximates
    # keeping the earliest encounter.
    sort_col = "encounter_id" if "encounter_id" in df.columns else id_col
    return df.sort_values(sort_col).drop_duplicates(subset=id_col, keep="first")


def split_columns(df: pd.DataFrame, cfg: dict):
    """Return (X_raw, y, groups) with a shared index.

    groups holds the readable sensitive columns used later for the fairness audit.
    """
    target = cfg["target"]["column"]
    drop_cols = [c for c in cfg["features"]["drop_columns"] if c in df.columns]

    y = df[target].astype(int)
    x_raw = df.drop(columns=drop_cols + [target], errors="ignore")

    group_cols = [c for c in cfg["fairness"]["group_columns"] if c in df.columns]
    groups = df[group_cols].copy()
    return x_raw, y, groups


def train_test_split_patient(x_raw, y, groups, df, cfg):
    """Split into train/test.

    If we deduplicated to one row per patient, a stratified split is already
    patient-safe. Otherwise we use a group split on the patient id so all of a
    patient's encounters stay on the same side.
    """
    test_size = cfg["split"]["test_size"]
    seed = cfg["split"]["random_state"]
    deduped = cfg["patient"].get("dedup_first_encounter", False)
    id_col = cfg["patient"]["id_column"]

    if deduped or id_col not in df.columns:
        return train_test_split(
            x_raw, y, groups, test_size=test_size, random_state=seed, stratify=y
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(x_raw, y, groups=df[id_col]))
    return (
        x_raw.iloc[train_idx], x_raw.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx],
    )


def build_preprocessor(x_raw: pd.DataFrame, cfg: dict) -> ColumnTransformer:
    """Impute missing values, then encode categoricals and scale numerics."""
    numeric_cols = x_raw.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in x_raw.columns if c not in numeric_cols]

    max_cats = cfg["features"].get("max_onehot_categories")

    numeric = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=max_cats,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
    )
