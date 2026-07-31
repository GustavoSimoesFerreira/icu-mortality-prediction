"""Streamlit demo for the ICU mortality project (hybrid: live + graceful fallback).

Run locally with Ollama for the full experience:
    streamlit run app/streamlit_app.py

If Ollama is not reachable (e.g. on Hugging Face Spaces), the risk model and
threshold analysis still work; the RAG and agent tabs show pre-generated
examples and explain how to enable live mode.
"""

from __future__ import annotations

import os
import sys

# Make `src` importable regardless of where Streamlit is launched from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from xgboost import XGBClassifier

from app.demo_utils import (
    EXAMPLE_AGENT,
    EXAMPLE_RAG,
    classify,
    metrics_at_threshold,
    ollama_available,
)
from src.features.preprocess import (
    apply_filters,
    build_preprocessor,
    dedup_first_encounter,
    split_columns,
    train_test_split_patient,
)

AGENT_CFG = os.path.join(REPO_ROOT, "configs", "agent.yaml")
MODEL_CFG = os.path.join(REPO_ROOT, "configs", "model.yaml")


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


@st.cache_resource(show_spinner="Loading model and data ...")
def load_context() -> dict:
    """Train the model once and prepare the held-out test set + SHAP explainer."""
    import shap

    with open(MODEL_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["input"]["processed_file"] = _abs(cfg["input"]["processed_file"])

    df = pd.read_parquet(cfg["input"]["processed_file"])
    df = apply_filters(df, cfg)
    df = dedup_first_encounter(df, cfg)
    x_raw, y, groups = split_columns(df, cfg)
    x_tr, x_te, y_tr, y_te, _g_tr, _g_te = train_test_split_patient(x_raw, y, groups, df, cfg)
    x_te = x_te.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)

    pre = build_preprocessor(x_tr, cfg)
    xt_tr = pre.fit_transform(x_tr)
    xt_te = pre.transform(x_te)
    pos = int(y_tr.sum())
    spw = (len(y_tr) - pos) / pos if pos else 1.0
    model = XGBClassifier(
        **cfg["model"]["xgboost"], scale_pos_weight=spw,
        eval_metric="logloss", tree_method="hist", random_state=cfg["split"]["random_state"],
    )
    model.fit(xt_tr, y_tr)
    probs = model.predict_proba(xt_te)[:, 1]

    explainer = shap.TreeExplainer(model)
    feature_names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    return {
        "x_te": x_te, "y_te": y_te, "xt_te": xt_te, "probs": probs,
        "explainer": explainer, "feature_names": feature_names,
    }


def load_agent_cfg() -> dict:
    with open(AGENT_CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_risk_tab(ctx: dict) -> None:
    st.subheader("Patient risk & explanation")
    n = len(ctx["x_te"])
    pid = st.number_input("Patient (test-set index)", 0, n - 1, 0, step=1)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.10, 0.01)

    prob = float(ctx["probs"][pid])
    label = classify(prob, threshold)
    c1, c2 = st.columns(2)
    c1.metric("Predicted 28-day mortality risk", f"{prob:.1%}")
    c2.metric("Classification at this threshold", label)

    m = metrics_at_threshold(ctx["y_te"], ctx["probs"], threshold)
    st.caption(
        f"At threshold {threshold:.2f}, the model captures {m['recall']:.0%} of "
        f"deaths (precision {m['precision']:.0%}) on the test set."
    )

    # Per-patient SHAP drivers.
    sv = np.asarray(ctx["explainer"].shap_values(ctx["xt_te"][pid : pid + 1]))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    row_sv = sv[0]
    order = np.argsort(-np.abs(row_sv))[:8]
    drivers = pd.DataFrame(
        {
            "factor": [ctx["feature_names"][i] for i in order],
            "impact_on_risk": [float(row_sv[i]) for i in order],
        }
    ).set_index("factor")
    st.markdown("**Top factors for this patient** (positive = increases risk):")
    st.bar_chart(drivers)


def render_evidence_tab(agent_cfg: dict, ollama_up: bool) -> None:
    st.subheader("Evidence assistant (RAG over PubMed)")
    if not ollama_up:
        st.info("Live mode needs a local Ollama server. Showing a pre-generated example.")
        st.markdown(f"**Q: {EXAMPLE_RAG['question']}**")
        st.write(EXAMPLE_RAG["answer"])
        st.caption("Sources: " + " | ".join(EXAMPLE_RAG["sources"]))
        return

    question = st.text_input("Ask a clinical question", EXAMPLE_RAG["question"])
    if st.button("Search evidence"):
        from src.rag.ask import answer, load_config
        rag_cfg = load_config(_abs(agent_cfg["rag_config"]))
        with st.spinner("Retrieving and generating ..."):
            reply, hits = answer(question, rag_cfg)
        st.write(reply)
        st.caption("Sources: " + " | ".join(f"[{i}] PMID {d['pmid']}" for i, (d, _) in enumerate(hits, 1)))


def render_agent_tab(agent_cfg: dict, ctx: dict, ollama_up: bool) -> None:
    st.subheader("Agentic clinical summary")
    if not ollama_up:
        st.info("Live mode needs a local Ollama server. Showing a pre-generated example.")
        st.markdown(f"**Patient {EXAMPLE_AGENT['patient']}**")
        st.write(EXAMPLE_AGENT["summary"])
        return

    pid = st.number_input("Patient for the agent", 0, len(ctx["x_te"]) - 1, 0, step=1, key="agent_pid")
    if st.button("Run agent"):
        from src.agent.agent import run_agent
        with st.spinner("The agent is reasoning and calling tools ..."):
            summary = run_agent(int(pid), agent_cfg, verbose=False)
        st.write(summary)


def main() -> None:
    st.set_page_config(page_title="ICU Mortality AI", layout="wide")
    st.title("ICU Mortality Prediction — risk model, evidence RAG, and an agent")
    st.caption("Research/portfolio demo. Not for clinical use. Data: PhysioNet (Open Access).")

    agent_cfg = load_agent_cfg()
    ollama_up = ollama_available(agent_cfg["agent"]["host"])
    status = "🟢 Ollama connected (live mode)" if ollama_up else "🟡 Ollama offline (example mode)"
    st.sidebar.markdown(f"**LLM status:** {status}")

    ctx = load_context()
    tab1, tab2, tab3 = st.tabs(["Risk & explanation", "Evidence assistant", "Agent"])
    with tab1:
        render_risk_tab(ctx)
    with tab2:
        render_evidence_tab(agent_cfg, ollama_up)
    with tab3:
        render_agent_tab(agent_cfg, ctx, ollama_up)


if __name__ == "__main__":
    main()
