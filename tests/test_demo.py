"""Offline tests for the demo helpers — no Streamlit, no network needed."""


from app.demo_utils import classify, metrics_at_threshold, ollama_available


def test_classify():
    assert classify(0.8, 0.5) == "HIGH RISK"
    assert classify(0.2, 0.5) == "lower risk"


def test_metrics_at_threshold():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    m = metrics_at_threshold(y, p, 0.5)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0


def test_ollama_unavailable_returns_false():
    # An unroutable port returns False quickly rather than raising.
    assert ollama_available("http://localhost:1", timeout=0.5) is False
