"""Coverage for probability calibration and the expected-value threshold it enables.

Calibration is the step that makes a model's output mean something: before it, the
average of a random forest's tree-vote share and a boosted model's imbalance-weighted
score is a number between 0 and 1 that is not the probability of anything. Two things
about it can fail quietly and are pinned here -- calibrating on too little data (which
produces confident nonsense rather than an obvious error), and forgetting to actually
use the calibrated members at prediction time.

The threshold logic on top of it has one failure mode worth stating outright, because
the first version of the script fell into it: the model predicts P(swing label), not
P(this trade wins), and those are wildly different events -- roughly 2% against 60% on
real data. Any rule that treats the model's probability as a win probability is wrong.
The rule that survives asks only that the probability *order* trades, and looks for the
point above which realized returns turn positive. These tests pin the three answers it
must give: a real floor, "no floor helps", and "the ordering is inverted".
"""
import os
import sys

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ensemble import (  # noqa: E402
    MIN_POSITIVES_FOR_CALIBRATION,
    MIN_POSITIVES_FOR_ISOTONIC,
    HybridSwingEnsemble,
    choose_calibration_method,
)
from scripts.expected_value_thresholds import (  # noqa: E402
    MIN_TRADES_PER_BIN,
    marginal_ev_curve,
    solve_threshold,
)


class _Overconfident(ClassifierMixin, BaseEstimator):
    """A classifier whose scores rank correctly but are far too extreme -- the distortion
    calibration exists to undo.

    Inherits the sklearn base classes because CalibratedClassifierCV refuses anything
    that doesn't identify itself as a classifier.
    """

    def __init__(self, sharpness=6.0):
        self.sharpness = sharpness

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_ = True
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        signal = np.asarray(X, dtype=float)[:, 0]
        positive = 1.0 / (1.0 + np.exp(-self.sharpness * (signal - 0.5)))
        return np.column_stack([1 - positive, positive])

    @property
    def feature_importances_(self):
        return np.array([1.0])


def _dataset(rows=1200, positive_rate=0.2, seed=0):
    """One informative feature; the label is drawn with probability equal to it, so the
    correctly-calibrated answer for a row with feature value v is exactly v."""
    rng = np.random.default_rng(seed)
    signal = rng.uniform(0, 1, rows)
    probability = positive_rate * 2 * signal
    labels = (rng.uniform(0, 1, rows) < probability).astype(int)
    return signal.reshape(-1, 1), labels


def _calibration_error(y_true, probabilities, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities > lower) & (probabilities <= upper)
        if in_bin.any():
            error += in_bin.sum() * abs(y_true[in_bin].mean() - probabilities[in_bin].mean())
    return error / len(y_true)


# -- calibration ---------------------------------------------------------------------


def test_method_choice_follows_the_positive_count():
    assert choose_calibration_method(np.ones(MIN_POSITIVES_FOR_ISOTONIC)) == "isotonic"
    assert choose_calibration_method(np.ones(MIN_POSITIVES_FOR_ISOTONIC - 1)) == "sigmoid"


def test_calibration_is_skipped_when_there_are_too_few_positives():
    """A mapping fitted on a handful of positives is confidently wrong, and downstream
    code treats a calibrated probability as a real one. Declining is the honest state."""
    X_fit, y_fit = _dataset(seed=1)
    X_calibrate = np.linspace(0, 1, 400).reshape(-1, 1)
    y_calibrate = np.zeros(400, dtype=int)
    y_calibrate[: MIN_POSITIVES_FOR_CALIBRATION - 1] = 1

    ensemble = HybridSwingEnsemble(_Overconfident(), _Overconfident(sharpness=3.0))
    ensemble.fit_calibrated(X_fit, y_fit, X_calibrate, y_calibrate)

    assert ensemble.is_calibrated is False
    assert ensemble.calibration_method is None


def test_calibration_is_skipped_when_the_slice_has_one_class():
    X_fit, y_fit = _dataset(seed=2)
    X_calibrate = np.linspace(0, 1, 400).reshape(-1, 1)
    ensemble = HybridSwingEnsemble(_Overconfident(), _Overconfident(sharpness=3.0))
    ensemble.fit_calibrated(X_fit, y_fit, X_calibrate, np.zeros(400, dtype=int))
    assert ensemble.is_calibrated is False


def test_calibration_reduces_the_gap_between_predictions_and_reality():
    X_fit, y_fit = _dataset(rows=3000, seed=3)
    X_calibrate, y_calibrate = _dataset(rows=3000, seed=4)
    X_report, y_report = _dataset(rows=3000, seed=5)

    ensemble = HybridSwingEnsemble(_Overconfident(), _Overconfident(sharpness=3.0))
    ensemble.fit(X_fit, y_fit)
    raw_error = _calibration_error(y_report, ensemble.predict_proba(X_report)[:, 1])

    ensemble.fit_calibrated(X_fit, y_fit, X_calibrate, y_calibrate)
    assert ensemble.is_calibrated is True
    calibrated_error = _calibration_error(y_report, ensemble.predict_proba(X_report)[:, 1])

    assert calibrated_error < raw_error


def test_predict_proba_uses_the_calibrated_members_once_fitted():
    X_fit, y_fit = _dataset(rows=2000, seed=6)
    X_calibrate, y_calibrate = _dataset(rows=2000, seed=7)
    ensemble = HybridSwingEnsemble(_Overconfident(), _Overconfident(sharpness=3.0))

    ensemble.fit(X_fit, y_fit)
    raw = ensemble.predict_proba(X_calibrate)[:, 1]
    ensemble.fit_calibrated(X_fit, y_fit, X_calibrate, y_calibrate)
    calibrated = ensemble.predict_proba(X_calibrate)[:, 1]

    assert not np.allclose(raw, calibrated), "calibrated members are fitted but not being used"
    assert calibrated.mean() == pytest.approx(y_calibrate.mean(), abs=0.05)


def test_probabilities_stay_in_range_and_rows_stay_aligned():
    X_fit, y_fit = _dataset(rows=1500, seed=8)
    X_calibrate, y_calibrate = _dataset(rows=1500, seed=9)
    ensemble = HybridSwingEnsemble(_Overconfident(), _Overconfident(sharpness=3.0))
    ensemble.fit_calibrated(X_fit, y_fit, X_calibrate, y_calibrate)

    probabilities = ensemble.predict_proba(X_calibrate)
    assert probabilities.shape == (len(X_calibrate), 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_feature_importances_still_come_from_the_base_estimators():
    """Calibration wraps outputs and has none of its own."""
    X_fit, y_fit = _dataset(rows=1200, seed=10)
    X_calibrate, y_calibrate = _dataset(rows=1200, seed=11)
    ensemble = HybridSwingEnsemble(_Overconfident(), _Overconfident(sharpness=3.0))
    ensemble.fit_calibrated(X_fit, y_fit, X_calibrate, y_calibrate)
    assert ensemble.feature_importances_.shape == (1,)


# -- the marginal expected-value curve ------------------------------------------------


def _trades(bands):
    """bands: [(count, mean_profit)] from lowest probability band upward."""
    probabilities, profits = [], []
    rng = np.random.default_rng(0)
    base = 0.0
    for count, mean in bands:
        probabilities.extend(np.linspace(base + 0.001, base + 0.05, count))
        draws = rng.normal(mean, 0.01, count)
        profits.extend(draws - draws.mean() + mean)
        base += 0.05
    return np.asarray(probabilities), np.asarray(profits)


class _NoScoring:
    """solve_threshold reads trades through simulate_trades; these tests drive the curve
    logic directly, so the detector/scoring pair is never consulted."""


def _solve(bands, monkeypatch):
    probabilities, profits = _trades(bands)
    monkeypatch.setattr(
        "scripts.expected_value_thresholds.null_trades",
        lambda detector, scored: (probabilities, profits),
    )
    return solve_threshold(_NoScoring(), {})


def test_curve_bins_by_quantile_so_every_bin_has_trades():
    probabilities, profits = _trades([(200, 0.01), (200, 0.02), (200, 0.03)])
    curve = marginal_ev_curve(probabilities, profits, bins=6)
    assert len(curve) >= 4
    assert all(band["num_trades"] > 0 for band in curve)
    assert sum(band["num_trades"] for band in curve) == len(probabilities)


def test_floor_is_found_when_the_bottom_band_loses_and_the_rest_win(monkeypatch):
    threshold, note, _ = _solve([(200, -0.02), (120, 0.03), (120, 0.04)], monkeypatch)
    assert threshold is not None and threshold > 0
    assert "non-negative in every bin" in note


def test_no_threshold_when_every_band_is_positive_from_the_bottom(monkeypatch):
    """Nothing is excluded, so the 'threshold' is the model declining to discriminate.
    Emitting 0 here would silently turn the app into an unconditional trader."""
    threshold, note, _ = _solve([(200, 0.02), (200, 0.025), (200, 0.03)], monkeypatch)
    assert threshold is None
    assert "no threshold excludes anything" in note


def test_no_threshold_when_the_ranking_is_inverted(monkeypatch):
    """Profit falls as predicted probability rises: the trades a floor keeps are the
    losing ones, and no floor can express that."""
    threshold, note, _ = _solve([(200, 0.04), (200, 0.01), (200, -0.05)], monkeypatch)
    assert threshold is None
    assert "inverted" in note


def test_no_threshold_when_the_whole_curve_loses(monkeypatch):
    threshold, note, _ = _solve([(200, -0.01), (200, -0.02), (200, -0.03)], monkeypatch)
    assert threshold is None
    assert "negative across the whole curve" in note


def test_too_few_trades_overall_yields_no_threshold(monkeypatch):
    """Quantile bins are equal-count, so a thin sample makes every band thin at once.
    A threshold resting on a dozen trades per band is noise dressed as a decision."""
    probabilities, profits = _trades([(20, -0.02), (20, 0.05), (20, 0.08)])
    curve = marginal_ev_curve(probabilities, profits, bins=8)
    assert all(band["num_trades"] < MIN_TRADES_PER_BIN for band in curve)

    threshold, note, _ = _solve([(20, -0.02), (20, 0.05), (20, 0.08)], monkeypatch)
    assert threshold is None
    assert "not estimable" in note
