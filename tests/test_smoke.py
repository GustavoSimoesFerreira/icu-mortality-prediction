"""Smoke tests — run offline (no network), keep CI green from day one."""

import pandas as pd

from src.data.load_diabetes import make_binary_target


def test_make_binary_target_maps_correctly():
    s = pd.Series(["<30", ">30", "NO", "<30"])
    result = make_binary_target(s, positive_value="<30")
    assert result.tolist() == [1, 0, 0, 1]


def test_make_binary_target_all_negative():
    s = pd.Series(["NO", ">30"])
    result = make_binary_target(s, positive_value="<30")
    assert result.sum() == 0
