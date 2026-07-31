"""Tools the agent can call: patient risk scoring and medical-evidence search.

- get_patient_risk: runs the trained risk model for one patient and returns the
  predicted 28-day mortality probability plus the top SHAP drivers for that
  patient (why the model thinks so).
- search_medical_evidence: retrieves relevant PubMed snippets from the RAG index.

The two tools connect Step 2 (the model) and Step 3 (the RAG index) so the
agent can reason across both.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from src.features.preprocess import (
    apply_filters,
    build_preprocessor,
    dedup_first_encounter,
    split_columns,
    train_test_split_patient,
)
from src.rag.ollama_client import OllamaClient
from src.rag.store import VectorStore


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean_name(name: str) -> str:
    """Strip the ColumnTransformer prefixes (num__/cat__) for readability."""
    return name.split("__", 1)[-1]


def _prepare_split(model_cfg: dict):
    """Reconstruct the deterministic train/test split (fixed seed)."""
    df = pd.read_parquet(model_cfg["input"]["processed_file"])
    df = apply_filters(df, model_cfg)
    df = dedup_first_encounter(df, model_cfg)
    x_raw, y, groups = split_columns(df, model_cfg)
    x_tr, x_te, y_tr, y_te, _g_tr, _g_te = train_test_split_patient(
        x_raw, y, groups, df, model_cfg
    )
    return x_tr.reset_index(drop=True), y_tr, x_te.reset_index(drop=True), y_te


def _load_or_train_model(agent_cfg: dict, model_cfg: dict) -> dict:
    """Load a cached fitted pipeline, or train and cache one."""
    model_path = Path(agent_cfg["risk"]["model_file"])
    if model_path.exists():
        return joblib.load(model_path)

    x_tr, y_tr, _x_te, _y_te = _prepare_split(model_cfg)
    pre = build_preprocessor(x_tr, model_cfg)
    xt_tr = pre.fit_transform(x_tr)
    xgb = XGBClassifier(
        **model_cfg["model"]["xgboost"],
        eval_metric="logloss",
        tree_method="hist",
        random_state=model_cfg["split"]["random_state"],
    )
    xgb.fit(xt_tr, y_tr)
    bundle = {"pre": pre, "model": xgb, "feature_names": list(pre.get_feature_names_out())}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def get_patient_risk(patient_id: int, agent_cfg: dict) -> dict:
    """Predict 28-day mortality risk and the top drivers for one test patient."""
    import shap

    model_cfg = _load_yaml(agent_cfg["model_config"])
    bundle = _load_or_train_model(agent_cfg, model_cfg)
    _x_tr, _y_tr, x_te, _y_te = _prepare_split(model_cfg)

    if patient_id < 0 or patient_id >= len(x_te):
        return {"error": f"patient_id must be between 0 and {len(x_te) - 1}"}

    row = x_te.iloc[[patient_id]]
    x = bundle["pre"].transform(row)
    prob = float(bundle["model"].predict_proba(x)[0, 1])

    explainer = shap.TreeExplainer(bundle["model"])
    sv = np.asarray(explainer.shap_values(x))
    if sv.ndim == 3:            # (n, features, classes)
        sv = sv[:, :, 1]
    row_sv = sv[0]

    order = np.argsort(-np.abs(row_sv))[:5]
    drivers = [
        {
            "factor": _clean_name(bundle["feature_names"][i]),
            "direction": "increases risk" if row_sv[i] > 0 else "decreases risk",
        }
        for i in order
    ]
    return {"patient_id": patient_id, "risk_28d_mortality": round(prob, 3), "top_drivers": drivers}


def search_medical_evidence(query: str, agent_cfg: dict) -> str:
    """Retrieve PubMed snippets relevant to a clinical query from the RAG index."""
    if not query or not query.strip():
        return "No query provided; cannot search for evidence."

    rag_cfg = _load_yaml(agent_cfg["rag_config"])
    store = VectorStore.load(rag_cfg["paths"]["index_file"], rag_cfg["paths"]["docs_file"])
    client = OllamaClient(
        rag_cfg["ollama"]["host"],
        rag_cfg["ollama"]["embed_model"],
        agent_cfg["agent"]["gen_model"],
        rag_cfg["ollama"]["timeout"],
    )
    query_vector = client.embed(query)
    hits = store.search(query_vector, k=rag_cfg["retrieval"]["top_k"])
    lines = []
    for i, (doc, _) in enumerate(hits, 1):
        lines.append(f"[{i}] {doc['title']} (PMID {doc['pmid']}): {doc['text'][:400]}")
    return "\n".join(lines) if lines else "No relevant evidence found."
