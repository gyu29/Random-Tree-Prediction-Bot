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
import pandas as pd
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
    """bands: [(count, mean_profit)] from lowest probability band upward.

    Entry dates are one per trade, strictly increasing, and spaced five calendar days
    apart, so every block holds independent trades and the run spans enough years to clear
    MIN_BLOCKS_FOR_RESAMPLING. The significance machinery then has no clustering to find
    and no block shortage to report, which is what these fixtures intend -- they are here
    to exercise the shape of the marginal curve, nothing else.
    """
    probabilities, profits = [], []
    rng = np.random.default_rng(0)
    base = 0.0
    for count, mean in bands:
        probabilities.extend(np.linspace(base + 0.001, base + 0.05, count))
        draws = rng.normal(mean, 0.01, count)
        profits.extend(draws - draws.mean() + mean)
        base += 0.05
    probabilities, profits = np.asarray(probabilities), np.asarray(profits)
    dates = (pd.Timestamp("2015-01-01", tz="UTC")
             + pd.to_timedelta(np.arange(len(profits)) * 5, unit="D")).to_numpy()
    order = np.argsort(probabilities)
    return probabilities[order], profits[order], dates


class _NoScoring:
    """solve_threshold reads trades through simulate_trades; these tests drive the curve
    logic directly, so the detector/scoring pair is never consulted."""


def _solve(bands, monkeypatch):
    probabilities, profits, dates = _trades(bands)
    monkeypatch.setattr(
        "scripts.expected_value_thresholds.null_trades",
        lambda detector, scored: (probabilities, profits, dates),
    )
    return solve_threshold(_NoScoring(), {})


def test_curve_bins_by_quantile_so_every_bin_has_trades():
    probabilities, profits, _ = _trades([(200, 0.01), (200, 0.02), (200, 0.03)])
    curve = marginal_ev_curve(probabilities, profits, bins=6)
    assert len(curve) >= 4
    assert all(band["num_trades"] > 0 for band in curve)
    assert sum(band["num_trades"] for band in curve) == len(probabilities)


def test_floor_is_found_when_the_bottom_band_loses_and_the_rest_win(monkeypatch):
    threshold, note, _ = _solve([(200, -0.02), (120, 0.03), (120, 0.04)], monkeypatch)
    assert threshold is not None and threshold > 0
    assert "standard errors" in note


def test_floor_is_found_when_bands_are_all_positive_but_clearly_separated(monkeypatch):
    """All bands profitable is not a reason to reject. An earlier rule required the bottom
    band to be *losing* money, which threw away any model whose trades were all profitable
    but very unevenly so -- growth_tech earns +1.34% above its floor against +0.64% below,
    a ranking that plainly works. What matters is separation, not the sign of the bottom."""
    threshold, note, _ = _solve([(200, 0.005), (200, 0.04), (200, 0.05)], monkeypatch)
    assert threshold is not None and threshold > 0
    assert "standard errors" in note


def test_no_threshold_when_bands_are_positive_but_indistinguishable(monkeypatch):
    """The case the old rule was reaching for and missed: every band profitable *and*
    flat, so a floor would exclude trades no worse than the ones it keeps."""
    threshold, note, _ = _solve([(200, 0.020), (200, 0.021), (200, 0.020)], monkeypatch)
    assert threshold is None
    assert "distinguishable from noise" in note


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
    probabilities, profits, _ = _trades([(20, -0.02), (20, 0.05), (20, 0.08)])
    curve = marginal_ev_curve(probabilities, profits, bins=8)
    assert all(band["num_trades"] < MIN_TRADES_PER_BIN for band in curve)

    threshold, note, _ = _solve([(20, -0.02), (20, 0.05), (20, 0.08)], monkeypatch)
    assert threshold is None
    assert "not estimable" in note


# -- the multiplicity correction ------------------------------------------------------


def test_permutation_bar_collapses_toward_a_single_test_with_one_candidate():
    """With nothing searched over there is nothing to pay for, so the bar should sit near
    an ordinary one-sided critical value rather than far above it."""
    from scripts.expected_value_thresholds import (  # noqa: F401
        block_bootstrap_se, date_blocks, permutation_critical_t,
    )

    rng = np.random.default_rng(1)
    profits = rng.normal(0.005, 0.05, 2000)
    probabilities = rng.uniform(0, 1, 2000)
    dates = pd.bdate_range("2010-01-01", periods=2000, tz="UTC").to_numpy()
    blocks = date_blocks(dates, block_length=10)
    candidate = float(np.quantile(probabilities, 0.5))
    errors = [block_bootstrap_se(profits, probabilities >= candidate, blocks, replicates=400)]
    bar = permutation_critical_t(profits, probabilities, [candidate], errors, blocks, 40,
                                 permutations=400)
    assert 1.2 < bar < 2.3, f"one candidate should cost roughly a single test, got {bar:.2f}"


def test_permutation_bar_rises_as_more_candidates_are_searched():
    """The search is not free: looking at more floors must raise the bar."""
    from scripts.expected_value_thresholds import (  # noqa: F401
        block_bootstrap_se, date_blocks, permutation_critical_t,
    )

    rng = np.random.default_rng(1)
    profits = rng.normal(0.005, 0.05, 2000)
    probabilities = rng.uniform(0, 1, 2000)
    dates = pd.bdate_range("2010-01-01", periods=2000, tz="UTC").to_numpy()
    blocks = date_blocks(dates, block_length=10)
    bars = []
    for count in (1, 8):
        candidates = list(np.quantile(probabilities, np.linspace(0.2, 0.8, count)))
        errors = [block_bootstrap_se(profits, probabilities >= c, blocks, replicates=400)
                  for c in candidates]
        bars.append(permutation_critical_t(profits, probabilities, candidates, errors, blocks,
                                           40, permutations=400))
    assert bars[1] > bars[0], f"searching 8 floors must cost more than 1: {bars}"


def test_block_bootstrap_widens_the_error_when_trades_are_clustered():
    """The reason for resampling blocks at all. Trades that arrive together and move
    together carry less information than their count suggests, and the standard error has
    to say so -- otherwise correlated bets are counted as independent evidence, which is
    what made seven of eight categories look like they had an edge."""
    from scripts.expected_value_thresholds import block_bootstrap_se, date_blocks

    rng = np.random.default_rng(3)
    days, per_day = 200, 5
    shared = rng.normal(0, 0.04, days)          # a common shock each day
    independent = rng.normal(0, 0.04, (days, per_day))
    clustered = (shared[:, None] + independent).ravel()
    scattered = rng.normal(0, 0.04, days * per_day)

    dates = np.repeat(pd.bdate_range("2010-01-01", periods=days, tz="UTC").to_numpy(), per_day)
    blocks = date_blocks(dates, block_length=10)
    mask = np.tile([True] * per_day, days).astype(bool)
    mask[: days * per_day // 2] = False

    clustered_se = block_bootstrap_se(clustered, mask, blocks, replicates=600)
    scattered_se = block_bootstrap_se(scattered, mask, blocks, replicates=600)
    assert clustered_se > scattered_se, (
        f"clustered returns must yield the larger error: {clustered_se:.5f} vs {scattered_se:.5f}"
    )


# -- the model-vs-null comparison ------------------------------------------------------


def _two_run_trades(years=9, seed=11):
    """A model run and a model-off null run over one shared price history.

    Both react to the same shocks and hold for six months, which is the dependence the
    paired bootstrap exists to preserve: they are not two independent samples of anything.
    Spans several years because at this horizon a block is a year wide.
    """
    rng = np.random.default_rng(seed)
    days, hold = years * 252, 126
    path = np.concatenate([[0.0], np.cumsum(rng.normal(0, 0.01, days + hold))])
    calendar = pd.bdate_range("2010-01-01", periods=days, tz="UTC").to_numpy()

    def run(count):
        entries = np.sort(rng.choice(days, size=count, replace=False))
        return path[entries + hold] - path[entries], calendar[entries]

    model_profits, model_dates = run(days // 12)
    null_profits, null_dates = run(days // 8)
    return model_profits, model_dates, null_profits, null_dates


def test_blocks_follow_the_calendar_not_the_count_of_entry_dates():
    """The defect this replaces, and it was silent. Blocks used to advance by 252 entries
    in the list of distinct entry dates; a six-month hold produces only a few dozen
    distinct entry dates across an entire test split, so every category collapsed to one
    block, every resample drew the identical trades, and the standard error came back
    0.0 -- read downstream as an edge of exactly zero rather than as no measurement."""
    from scripts.expected_value_thresholds import date_blocks

    # 30 distinct entry dates, heavily clustered, spread across eight years.
    dates = np.repeat(
        pd.to_datetime(["2010-03-01", "2011-06-01", "2012-09-01", "2013-01-15", "2014-07-01",
                        "2015-11-01", "2016-04-01", "2017-08-01", "2018-02-01", "2019-05-01"]).to_numpy(),
        12,
    )
    blocks = date_blocks(dates)
    assert len(blocks) >= 6, (
        f"ten dates across nine years must not be one block: got {len(blocks)}"
    )
    assert sum(len(b) for b in blocks) == len(dates), "every trade belongs to exactly one block"


def test_too_few_blocks_reports_no_measurement_rather_than_no_error():
    """A standard error of 0.0 is a claim of perfect precision, and it arrives exactly
    when precision is absent. Callers must be handed NaN so they can say so."""
    from scripts.expected_value_thresholds import (
        block_bootstrap_difference_se,
        block_bootstrap_se,
        paired_date_blocks,
    )

    model_p, model_d, null_p, null_d = _two_run_trades(years=2)
    blocks = paired_date_blocks(model_d, null_d)
    assert len(blocks) < 6, "fixture is meant to be too short to resample"
    assert np.isnan(block_bootstrap_difference_se(model_p, null_p, blocks, replicates=200))
    assert np.isnan(block_bootstrap_se(model_p, np.arange(len(model_p)) % 2 == 0,
                                       [np.arange(len(model_p))], replicates=200))


def test_paired_blocks_keep_both_runs_on_one_partition():
    """Every trade from both runs lands in exactly one block, indexed off a shared origin.
    Partitioned separately, a block kind to the model and the same block kind to the null
    would resample independently and their shared luck would count twice."""
    from scripts.expected_value_thresholds import paired_date_blocks

    model_p, model_d, null_p, null_d = _two_run_trades()
    blocks = paired_date_blocks(model_d, null_d)

    assert sorted(np.concatenate([b[0] for b in blocks])) == list(range(len(model_p)))
    assert sorted(np.concatenate([b[1] for b in blocks])) == list(range(len(null_p)))


def test_an_edge_of_zero_stays_inside_two_standard_errors():
    """Calibration check. A resampler that only ever widened its error would pass every
    other test here by reporting infinity; when the two runs differ by nothing but noise,
    the measured edge has to sit inside its own error."""
    from scripts.expected_value_thresholds import (
        block_bootstrap_difference_se,
        paired_date_blocks,
    )

    for seed in (5, 11, 23):
        model_p, model_d, null_p, null_d = _two_run_trades(seed=seed)
        error = block_bootstrap_difference_se(
            model_p, null_p, paired_date_blocks(model_d, null_d), replicates=800
        )
        edge = float(np.mean(model_p) - np.mean(null_p))
        assert abs(edge) < 2 * error, f"seed {seed}: edge {edge:+.4f} vs SE {error:.4f}"


def test_a_handful_of_model_trades_cannot_report_an_edge():
    """The failure this guard exists for. At the deployed floor one category took a single
    trade against its null's 142, and the paired resample called it +12.75 standard errors:
    almost every replicate was discarded for containing no model trade, and the agreement
    of the few survivors read as precision. Blocks where only one side traded carry no
    difference and must not be counted toward the minimum."""
    from scripts.expected_value_thresholds import (
        block_bootstrap_difference_se,
        paired_date_blocks,
    )

    _, _, null_p, null_d = _two_run_trades()
    model_p, model_d = null_p[:1], null_d[:1]
    blocks = paired_date_blocks(model_d, null_d)

    assert len(blocks) >= 6, "the null alone spans plenty of blocks -- that is the trap"
    assert np.isnan(block_bootstrap_difference_se(model_p, null_p, blocks, replicates=400))
