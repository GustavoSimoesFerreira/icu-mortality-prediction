"""Offline test for the tuning module — no data or model needed."""

from src.models.tune import build_search_space


def test_build_search_space_prefixes_and_adds_weight():
    tcfg = {"search_space": {"n_estimators": [100, 200], "max_depth": [3, 4]}}
    space = build_search_space(tcfg, scale_pos_weight=5.0)
    assert "clf__n_estimators" in space
    assert "clf__max_depth" in space
    # class-imbalance weighting is offered as a tunable choice
    assert space["clf__scale_pos_weight"] == [1.0, 5.0]
    # original keys are not left unprefixed
    assert "n_estimators" not in space
