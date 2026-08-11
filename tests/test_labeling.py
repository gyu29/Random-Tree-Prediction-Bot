"""Row-for-row equivalence check: vectorized create_swing_labels vs. the original
Python for-loop it replaces. The original loop (kept here verbatim, not imported,
since the module it lived in is being retired) is the ground truth for *shape* of
the computation; this test exists so the vectorized rewrite in app/labeling.py
can't silently drift from it.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.labeling import create_swing_labels, effective_threshold  # noqa: E402


def _reference_loop_labels(df, swing_threshold, lookforward_periods, min_hold_periods):
    """Verbatim port of the original row-by-row loop, with only the threshold
    fixed to match effective_threshold() (the rest of the arithmetic is untouched)."""
    df = df.copy()
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    threshold = effective_threshold(swing_threshold)
    df["swing_label"] = 0
    df["swing_profit_potential"] = 0.0
    df["swing_risk"] = 0.0
    for i in range(len(df) - lookforward_periods):
        current_price = df.iloc[i][price_col]
        future_slice = df.iloc[i + min_hold_periods : i + lookforward_periods + 1]
        if len(future_slice) > 0:
            max_future_high = future_slice["high"].max()
            min_future_low = future_slice["low"].min()
            upside_potential = (max_future_high - current_price) / current_price
            downside_risk = (current_price - min_future_low) / current_price
            if upside_potential >= threshold:
                df.iloc[i, df.columns.get_loc("swing_label")] = 1
                df.iloc[i, df.columns.get_loc("swing_profit_potential")] = upside_potential
                df.iloc[i, df.columns.get_loc("swing_risk")] = downside_risk
    return df


def _make_sample(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "adj_close": close, "volume": volume},
        index=dates,
    )


def check_equivalence(swing_threshold, lookforward_periods, min_hold_periods, n=600, seed=0):
    df = _make_sample(n=n, seed=seed)
    expected = _reference_loop_labels(df, swing_threshold, lookforward_periods, min_hold_periods)
    actual = create_swing_labels(df, swing_threshold, lookforward_periods, min_hold_periods)

    label_match = (expected["swing_label"] == actual["swing_label"]).all()
    profit_match = np.allclose(expected["swing_profit_potential"], actual["swing_profit_potential"], atol=1e-9)
    risk_match = np.allclose(expected["swing_risk"], actual["swing_risk"], atol=1e-9)

    ok = bool(label_match and profit_match and risk_match)
    detail = (
        f"threshold={swing_threshold} lookforward={lookforward_periods} min_hold={min_hold_periods} n={n}: "
        f"label_match={label_match} profit_match={profit_match} risk_match={risk_match} "
        f"positive_count={int(actual['swing_label'].sum())}"
    )
    return ok, detail


CASES = [
    dict(swing_threshold=0.15, lookforward_periods=10, min_hold_periods=3),
    dict(swing_threshold=0.08, lookforward_periods=10, min_hold_periods=3),
    dict(swing_threshold=0.15, lookforward_periods=5, min_hold_periods=1),
    dict(swing_threshold=0.15, lookforward_periods=20, min_hold_periods=3, n=800),
    dict(swing_threshold=0.15, lookforward_periods=3, min_hold_periods=3),  # window_size == 1
    dict(swing_threshold=0.30, lookforward_periods=10, min_hold_periods=3, seed=1),
]


def run_all():
    all_ok = True
    for case in CASES:
        ok, detail = check_equivalence(**case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {detail}")
        all_ok = all_ok and ok
    return all_ok


def test_vectorized_labels_match_reference_loop():
    """pytest entry point for the same cases run_all() prints -- kept as plain asserts (no
    pytest import) so `python tests/test_labeling.py` still works standalone, with no new
    dependency, while `pytest` also collects and runs this rather than silently finding 0
    tests (nothing here was named test_* before)."""
    for case in CASES:
        ok, detail = check_equivalence(**case)
        assert ok, detail


if __name__ == "__main__":
    success = run_all()
    print("\nALL PASS" if success else "\nSOME FAILED")
    sys.exit(0 if success else 1)
