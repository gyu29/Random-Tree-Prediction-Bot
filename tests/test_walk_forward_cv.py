"""Coverage for purged walk-forward cross-validation.

Two things here can silently produce numbers that look fine and mean nothing, so both
are pinned:

  * Fold geometry. Folds must expand rather than slide (each one trains on all history
    before its validation block, the shape deployment actually has), must not overlap,
    must carry the embargo at every boundary, and must never read test/. A fold that
    trains on the future produces excellent metrics and no information.
  * The selection rule. Pooling five folds and then applying the single-window trade
    floor would be a *weaker* bar than not cross-validating at all -- 20 pooled trades
    is four per fold -- so the floor scales with the fold count, and a candidate must
    also be positive in a majority of the folds it traded in. Aggregate-only selection
    cannot tell "works everywhere" from "spectacular in one regime, negative in four",
    which is the failure cross-validation exists to catch.

Fold geometry is checked on synthetic frames; the selection rule is checked directly on
constructed sweeps, so neither test fits a model.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DEFAULT_LOOKFORWARD_PERIODS  # noqa: E402
from scripts.build_factor_datasets import split_by_cutoffs  # noqa: E402
from scripts.walk_forward_cv import (  # noqa: E402
    FOLD_AGREEMENT,
    MIN_TRADES_PER_FOLD,
    make_fold_cutoffs,
    select_across_folds,
    summarize,
)

TRADING_DAYS = pd.bdate_range("2000-01-03", periods=5000, tz="UTC")


def _frame(start, rows, seed=0):
    rng = np.random.default_rng(seed)
    index = TRADING_DAYS[start:start + rows]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(index))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "adj_close": close,
         "volume": rng.integers(1_000_000, 3_000_000, len(index))},
        index=index,
    )


def _category(n_symbols=5, end=4600):
    return {f"SYM{i}": _frame(i * 400, end - i * 400, seed=i) for i in range(n_symbols)}


# -- fold geometry ------------------------------------------------------------------


def test_folds_expand_rather_than_slide():
    """Each fold must train on everything before its validation block. A sliding window
    would discard history the deployed model would have had."""
    cutoffs = make_fold_cutoffs(_category(), n_folds=5)
    assert len(cutoffs) == 5
    train_ends = [train_end for train_end, _ in cutoffs]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends), "folds must advance, not repeat"
    for train_end, validation_end in cutoffs:
        assert train_end < validation_end


def test_fold_validation_blocks_do_not_overlap():
    cutoffs = make_fold_cutoffs(_category(), n_folds=5)
    for (_, earlier_end), (later_start, _) in zip(cutoffs, cutoffs[1:]):
        assert later_start >= earlier_end, "one fold validates on another fold's block"


def test_every_fold_boundary_carries_the_embargo():
    """A fold cut without a gap trains on rows whose labels are decided by bars inside
    its own validation block."""
    histories = _category()
    for train_end, validation_end in make_fold_cutoffs(histories, n_folds=5):
        for frame in histories.values():
            parts = split_by_cutoffs(frame, train_end, validation_end)
            if parts["train"].empty or parts["validation"].empty:
                continue
            positions = {date: i for i, date in enumerate(frame.index)}
            gap = positions[parts["validation"].index.min()] - positions[parts["train"].index.max()]
            assert gap > DEFAULT_LOOKFORWARD_PERIODS


def test_folds_stay_inside_the_supplied_history():
    """Cutoffs are dates from the pooled calendar, so no fold can reach past the data it
    was handed -- which is what keeps test/ out of cross-validation, since the caller
    only ever passes train/ + validation/."""
    histories = _category()
    latest = max(frame.index.max() for frame in histories.values())
    earliest = min(frame.index.min() for frame in histories.values())
    for train_end, validation_end in make_fold_cutoffs(histories, n_folds=5):
        assert earliest <= train_end <= latest
        assert earliest <= validation_end <= latest


def test_fold_count_is_respected():
    for n_folds in (3, 5, 8):
        assert len(make_fold_cutoffs(_category(), n_folds=n_folds)) == n_folds


# -- the selection rule --------------------------------------------------------------


def _sweep_row(threshold, profits):
    return {"threshold": threshold, **summarize(profits)}


def _uniform(mean, count):
    """`count` trades whose mean is `mean`, with enough spread to have a real standard
    error -- a zero-variance sample would make the shrinkage vanish."""
    rng = np.random.default_rng(int(abs(mean) * 1e6) + count)
    draws = rng.normal(mean, 0.02, count)
    return list(draws - draws.mean() + mean)


def test_trade_floor_scales_with_the_number_of_folds():
    """40 pooled trades passes a single-window floor of 20 and must fail a five-fold one."""
    profits_per_fold = [_uniform(0.03, 8) for _ in range(5)]  # 40 trades, 8 per fold
    pooled = [p for fold in profits_per_fold for p in fold]
    sweep = [_sweep_row(0.30, _uniform(0.001, 400)),
             _sweep_row(0.35, pooled),
             _sweep_row(0.40, _uniform(0.001, 400))]
    per_fold = {0.30: [_uniform(0.001, 80) for _ in range(5)],
                0.35: profits_per_fold,
                0.40: [_uniform(0.001, 80) for _ in range(5)]}

    assert len(pooled) > MIN_TRADES_PER_FOLD, "fixture must clear the per-fold floor to be meaningful"
    selected, annotated, note = select_across_folds(sweep, per_fold, default_threshold=0.65)
    row = next(r for r in annotated if r["threshold"] == 0.35)
    assert row["trade_floor"] == MIN_TRADES_PER_FOLD * 5
    assert row["eligible"] is False
    assert selected == 0.65 and "kept the deployed value" in note


def test_candidate_good_in_one_fold_and_negative_in_the_rest_is_rejected():
    """The whole reason for cross-validating: a pooled mean can be carried by one regime."""
    lopsided = [_uniform(0.50, 40)] + [_uniform(-0.02, 40) for _ in range(4)]
    pooled = [p for fold in lopsided for p in fold]
    assert np.mean(pooled) > 0, "fixture must look good in aggregate, or it proves nothing"

    sweep = [_sweep_row(0.30, _uniform(-0.05, 200)),
             _sweep_row(0.35, pooled),
             _sweep_row(0.40, _uniform(-0.05, 200))]
    per_fold = {0.30: [_uniform(-0.05, 40) for _ in range(5)],
                0.35: lopsided,
                0.40: [_uniform(-0.05, 40) for _ in range(5)]}

    _, annotated, _ = select_across_folds(sweep, per_fold, default_threshold=0.65)
    row = next(r for r in annotated if r["threshold"] == 0.35)
    assert row["agreeing_folds"] == 1
    assert row["folds_needed"] == int(np.ceil(FOLD_AGREEMENT * 5))
    assert row["eligible"] is False, "aggregate-positive but wrong in 4 of 5 folds must not qualify"


def test_candidate_consistent_across_folds_is_selected():
    consistent = [_uniform(0.03, 40) for _ in range(5)]
    weaker = [_uniform(0.005, 40) for _ in range(5)]
    sweep = [_sweep_row(0.30, [p for f in weaker for p in f]),
             _sweep_row(0.35, [p for f in consistent for p in f]),
             _sweep_row(0.40, [p for f in weaker for p in f])]
    per_fold = {0.30: weaker, 0.35: consistent, 0.40: weaker}

    selected, annotated, note = select_across_folds(sweep, per_fold, default_threshold=0.65)
    row = next(r for r in annotated if r["threshold"] == 0.35)
    assert row["agreeing_folds"] == 5
    assert selected == 0.35
    assert "interior peak" in note


def test_selection_still_requires_an_interior_peak():
    """Inherited from the single-window rule: a maximum against a constraint is the
    constraint. Here the best score sits at the grid edge."""
    rising = {0.30: 0.01, 0.35: 0.02, 0.40: 0.04}
    per_fold = {t: [_uniform(m, 40) for _ in range(5)] for t, m in rising.items()}
    sweep = [_sweep_row(t, [p for f in per_fold[t] for p in f]) for t in sorted(rising)]

    selected, _, note = select_across_folds(sweep, per_fold, default_threshold=0.65)
    assert selected == 0.65
    assert "edge of the swept grid" in note


def test_no_eligible_candidate_keeps_the_deployed_value():
    losing = {t: [_uniform(-0.03, 40) for _ in range(5)] for t in (0.30, 0.35, 0.40)}
    sweep = [_sweep_row(t, [p for f in losing[t] for p in f]) for t in sorted(losing)]
    selected, _, note = select_across_folds(sweep, losing, default_threshold=0.45)
    assert selected == 0.45
    assert "no candidate survived cross-validation" in note


def test_summarize_penalises_a_thin_sample():
    """The shrunk score is what selection ranks on: same mean, fewer trades, lower score."""
    thick = summarize(_uniform(0.02, 500))
    thin = summarize(_uniform(0.02, 6))
    assert thick["mean_profit"] == pytest.approx(thin["mean_profit"])
    assert thick["score"] > thin["score"]
    assert summarize([])["num_trades"] == 0
