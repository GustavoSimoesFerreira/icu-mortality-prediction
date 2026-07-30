"""
SHAP explanations for the tree model.

In healthcare, a model has to be interpretable to be trusted. SHAP shows which
features drive each prediction; the summary plot ranks features by overall impact.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so it works in CI / servers
import matplotlib.pyplot as plt
import numpy as np
import shap


def save_shap_summary(model, x_transformed, feature_names, out_path, sample_size=2000):
    """Compute SHAP values on a sample and save a beeswarm summary plot."""
    x = np.asarray(x_transformed)
    if sample_size and len(x) > sample_size:
        rng = np.random.default_rng(42)
        x = x[rng.choice(len(x), size=sample_size, replace=False)]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values, x, feature_names=list(feature_names), show=False, max_display=15
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
