"""Pure helpers for the Streamlit demo (no Streamlit import, so unit-testable).

Kept separate from the UI so the logic can be tested without a running app.
"""

from __future__ import annotations

import numpy as np
import requests
from sklearn.metrics import confusion_matrix


def ollama_available(host: str, timeout: float = 2.0) -> bool:
    """Return True if a local Ollama server responds, else False.

    Used to switch the demo between live LLM mode and pre-generated examples.
    """
    try:
        requests.get(host, timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False


def classify(prob: float, threshold: float) -> str:
    return "HIGH RISK" if prob >= threshold else "lower risk"


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Recall and precision on the test set at a given decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    _tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"recall": round(float(recall), 3), "precision": round(float(precision), 3)}


# --- Pre-generated examples shown when Ollama is not available (e.g. on Spaces) ---

EXAMPLE_RAG = {
    "question": "What clinical scores predict ICU mortality?",
    "answer": (
        "According to the sources, several validated scores predict ICU mortality: "
        "APACHE II [1], SAPS III [2], and SOFA, SAPS II and OASIS [3]. These combine "
        "physiological measurements to estimate severity of illness."
    ),
    "sources": [
        (
            "[1] ICU-specific determinants of mortality in critically ill patients "
            "with infective endocarditis (PMID 42157104)"
        ),
        (
            "[2] Improving mortality prediction in critically ill cancer patients "
            "(PMID 42384111)"
        ),
        (
            "[3] An explainable ML model for mortality prediction in ICU lung-cancer "
            "patients (PMID 42445719)"
        ),
    ],
}

EXAMPLE_AGENT = {
    "patient": 0,
    "summary": (
        "The predicted 28-day mortality risk for patient 0 is low, with the main driver "
        "being an elevated admission pCO2, which increases mortality risk [1]. The "
        "literature indicates that elevated arterial CO2 tension is associated with poor "
        "outcomes in critically ill patients, reflecting respiratory or circulatory "
        "dysfunction [1]."
    ),
}
