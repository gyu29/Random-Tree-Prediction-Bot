"""Coverage for the gate that stops a model which failed its out-of-sample check from
being presented as a trading signal.

small_cap's trained model scored -0.43%/trade against its own null on test/ -- worse
than ignoring it and trading every eligible bar -- so its probability is not a number
anybody should act on. It is listed in app.config.CATEGORIES_FAILING_VALIDATION.

The gate warns rather than refuses, and the distinction is the thing worth testing:
scoring still works, because investigating a broken model requires being able to run
it. What stops is the presentation of that output as actionable -- no alert, no rank in
the screener, and a warning on every payload so no consumer can render the probability
without having been told.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import detector as detector_module  # noqa: E402
from app.detector import SwingTradeDetector  # noqa: E402
from app.trading_system import _screen_sort_key  # noqa: E402

TRUSTED = "widgets_ok"
UNTRUSTED = "widgets_failed"
WARNING = "measured worse than its own null"


@pytest.fixture(autouse=True)
def gated_categories(monkeypatch):
    """A synthetic gate list, so these tests describe the mechanism rather than pinning
    whichever categories happen to be failing today."""
    monkeypatch.setattr(detector_module, "CATEGORIES_FAILING_VALIDATION", {UNTRUSTED: WARNING})


class _FakeModel:
    def __init__(self, probability=0.9):
        self.probability = probability
        self.decision_threshold = 0.5

    def predict(self, X):
        return np.array([int(self.probability >= self.decision_threshold)] * len(X))

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.probability), np.full(len(X), self.probability)])


class _FakeScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def _loaded_detector(category, probability=0.9):
    """A detector wired past model_registry.load -- these tests are about the gate, not
    about artifact loading, which tests/test_model_registry.py already covers."""
    built = SwingTradeDetector.__new__(SwingTradeDetector)
    built.category = category
    built.provider = None
    built.validation_warning = detector_module.CATEGORIES_FAILING_VALIDATION.get(category)
    built.model = _FakeModel(probability)
    built.scaler = _FakeScaler()
    built.feature_columns = ["rsi_14"]
    built.training_stats = {}
    built.manifest = None
    built.version_warnings = []
    built.swing_threshold = 0.08
    built.effective_swing_threshold = 0.08
    built.lookforward_periods = 10
    built.decision_threshold = 0.5
    built._load_error = None
    return built


def _ohlcv(rows=150):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adj_close": close,
            "volume": rng.integers(1_000_000, 2_000_000, rows),
        },
        index=pd.bdate_range("2020-01-01", periods=rows),
    )


def test_trusted_category_has_no_warning_and_is_trustworthy():
    built = _loaded_detector(TRUSTED)
    assert built.validation_warning is None
    assert built.is_trustworthy is True


def test_gated_category_is_not_trustworthy_but_still_loads():
    built = _loaded_detector(UNTRUSTED)
    assert built.is_ready is True, "the gate must not prevent the model from loading"
    assert built.is_trustworthy is False
    assert built.validation_warning == WARNING


def test_gated_category_still_produces_a_prediction():
    """Refusing to score would make a broken model impossible to investigate."""
    result = _loaded_detector(UNTRUSTED).detect_swing_opportunity(_ohlcv(), "AAA")
    assert result["swing_probability"] == pytest.approx(0.9)
    assert result["source"] == "model"


def test_every_payload_from_a_gated_category_carries_the_warning():
    result = _loaded_detector(UNTRUSTED).detect_swing_opportunity(_ohlcv(), "AAA")
    assert result["validation_warning"] == WARNING


def test_payload_from_a_trusted_category_carries_no_warning():
    result = _loaded_detector(TRUSTED).detect_swing_opportunity(_ohlcv(), "AAA")
    assert result["validation_warning"] is None
    # Present as a key either way, so a consumer can read it unconditionally.
    assert "validation_warning" in result


def test_untrusted_rows_sort_below_every_ranked_row_whatever_their_probability():
    """The screener returns a ranking, and a ranking is a recommendation. A gated model's
    row keeps its probability but must never outrank a trustworthy one."""
    rows = [
        {"symbol": "LOW", "status": "ok", "swing_probability": 0.01},
        {"symbol": "GATED", "status": "untrusted_model", "swing_probability": 0.99},
        {"symbol": "HIGH", "status": "ok", "swing_probability": 0.80},
        {"symbol": "GONE", "status": "unavailable"},
    ]
    ordered = [row["symbol"] for row in sorted(rows, key=_screen_sort_key, reverse=True)]
    assert ordered[:2] == ["HIGH", "LOW"]
    assert set(ordered[2:]) == {"GATED", "GONE"}


def test_untrusted_row_is_not_dropped_from_the_screener():
    """Sorted last, not removed -- the symbol still exists and the user should see it,
    with status explaining why it has no rank."""
    rows = [
        {"symbol": "GATED", "status": "untrusted_model", "swing_probability": 0.99},
        {"symbol": "OK", "status": "ok", "swing_probability": 0.10},
    ]
    ordered = sorted(rows, key=_screen_sort_key, reverse=True)
    assert len(ordered) == 2
    assert ordered[-1]["symbol"] == "GATED"
    assert ordered[-1]["status"] == "untrusted_model"


def test_gate_list_entries_explain_themselves():
    """A bare category name in the gate would leave the UI with nothing to display and
    the next reader with no way to know why it was added."""
    from app.config import CATEGORIES_FAILING_VALIDATION

    for category, reason in CATEGORIES_FAILING_VALIDATION.items():
        assert isinstance(reason, str) and len(reason) > 40, (
            f"{category} needs a reason a user can read, not just a listing"
        )
