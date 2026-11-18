"""Tests de la phase 5 : métriques et intervalle de confiance."""
import numpy as np
import pytest

from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics


def test_perfect_predictions_score_one():
    y = np.array(["a", "b", "c", "a"])
    m = classification_metrics(y, y)
    assert m["n"] == 4 and m["accuracy"] == m["macro_f1"] == m["balanced_accuracy"] == pytest.approx(1)


def test_macro_f1_weights_classes_equally():
    y_true = np.array(["a"] * 9 + ["b"])
    y_pred = np.array(["a"] * 10)                     # ignore la classe rare
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.9) and m["macro_f1"] < 0.5


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y_true = rng.choice(["a", "b", "c"], 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, rng.choice(["a", "b", "c"], 300))
    groups = np.repeat(np.arange(100), 3)            # 3 régions par image
    low, high = bootstrap_macro_f1(y_true, y_pred, groups, n_boot=200)
    point = classification_metrics(y_true, y_pred)["macro_f1"]
    assert low < point < high
    assert (low, high) == bootstrap_macro_f1(y_true, y_pred, groups, n_boot=200)
