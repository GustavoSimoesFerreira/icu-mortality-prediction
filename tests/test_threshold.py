"""Offline tests for the threshold analysis — no model or data needed."""

import json

import numpy as np

from src.evaluation.threshold_analysis import recommend_threshold, sweep_thresholds


def test_sweep_and_recommend_meets_target():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    rows = sweep_thresholds(y, p, [0.2, 0.5, 0.7])
    rec = recommend_threshold(rows, target_recall=1.0)
    # thresholds 0.2 and 0.5 both give recall 1.0; the higher one (0.5) wins
    assert rec["threshold"] == 0.5
    assert rec["recall"] == 1.0


def test_recommend_returns_none_when_unreachable():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    rows = sweep_thresholds(y, p, [0.95])  # recall 0 at this threshold
    assert recommend_threshold(rows, target_recall=0.9) is None


def test_results_are_json_serializable():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.7, 0.3, 0.8])
    rows = sweep_thresholds(y, p, [0.5])
    json.dumps(rows)  # must not raise (no numpy types)
