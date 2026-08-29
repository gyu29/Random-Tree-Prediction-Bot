"""Coverage for the two split boundaries and the embargo each one carries.

There are two independent cuts in this project, and both were changed together
because both had the same class of defect -- information from after a boundary
reaching a model fitted before it:

  1. The dataset split (scripts/build_factor_datasets.py): two calendar cutoffs per
     category, shared by every symbol in it. The property that matters is
     category-wide, not per-symbol -- no symbol's test rows may fall inside any other
     symbol's training window. The superseded per-symbol percentage split is kept
     verbatim below as `_legacy_per_symbol_split`, the way tests/test_labeling.py
     keeps the loop its vectorized labeler replaced, so the leakage test can show it
     catches the thing it was written for rather than passing vacuously.

  2. The internal split inside app/trainer.py, which now cuts three slices per symbol
     -- fit, calibrate, report -- with an embargo at both boundaries. Three disjoint
     slices matter because reusing one for two purposes (calibrating on the rows the
     metrics are then reported on) makes the reported number optimistic by exactly the
     amount that matters. Its masks are positional; they used to be index-label based,
     and because symbols share dates, holding out one symbol's recent rows also held out
     every other symbol's rows on those dates.
     `test_internal_split_is_positional_not_label_based` is the regression test.

Everything here runs on synthetic frames -- no network, no yfinance call, and nothing
that reads or writes the real train/ validation/ test/ directories.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DEFAULT_LOOKFORWARD_PERIODS  # noqa: E402
from app.trainer import (  # noqa: E402
    INTERNAL_CALIBRATION_FRACTION,
    INTERNAL_VALIDATION_FRACTION,
    SwingTradeTrainer,
)
from scripts.build_factor_datasets import (  # noqa: E402
    EMBARGO_PERIODS,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    category_cutoffs,
    split_by_cutoffs,
)

TRADING_DAYS = pd.bdate_range("2000-01-03", periods=6000, tz="UTC")


def _ohlcv(start_position, rows, seed=0):
    """A price series on a slice of the shared TRADING_DAYS calendar. Symbols built
    from overlapping slices are what a factor category actually looks like: several
    wrappers around one asset, each listed in a different year."""
    rng = np.random.default_rng(seed)
    index = TRADING_DAYS[start_position:start_position + rows]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(index))))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.004, len(index))),
            "high": close * (1 + np.abs(rng.normal(0, 0.008, len(index)))),
            "low": close * (1 - np.abs(rng.normal(0, 0.008, len(index)))),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(index)),
        },
        index=index,
    )


def _staggered_category():
    """A category shaped like market_beta: one very long history plus six shorter ones
    that start later and run to the same recent end. This is the layout the per-symbol
    percentage split handled badly."""
    end = 5800
    return {
        "LONG": _ohlcv(0, end, seed=1),          # the ^GSPC-shaped outlier
        "MID_A": _ohlcv(800, end - 800, seed=2),
        "MID_B": _ohlcv(1200, end - 1200, seed=3),
        "MID_C": _ohlcv(1800, end - 1800, seed=4),
        "LATE_A": _ohlcv(3000, end - 3000, seed=5),
        "LATE_B": _ohlcv(3600, end - 3600, seed=6),
        "LATE_C": _ohlcv(4200, end - 4200, seed=7),
    }


def _legacy_per_symbol_split(df):
    """The superseded split: each symbol's own rows cut at 55%/70% of *its* length.
    Kept verbatim so test_no_test_row_falls_in_another_symbols_train_window can assert
    this leaks on the same fixture the calendar split handles cleanly."""
    df = df.sort_index()
    train_end = int(len(df) * TRAIN_FRACTION)
    validation_end = int(len(df) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return {
        "train": df.iloc[:train_end],
        "validation": df.iloc[train_end:validation_end],
        "test": df.iloc[validation_end:],
    }


def _cross_symbol_leak_fraction(frames_by_symbol):
    """Share of test rows sitting on a date inside some *other* symbol's train window.
    The number the calendar split exists to drive to zero."""
    train_spans = {
        symbol: (frames["train"].index.min(), frames["train"].index.max())
        for symbol, frames in frames_by_symbol.items()
        if not frames["train"].empty
    }
    covered = total = 0
    for symbol, frames in frames_by_symbol.items():
        test_index = frames["test"].index
        if test_index.empty:
            continue
        leaked = np.zeros(len(test_index), dtype=bool)
        for other, (first, last) in train_spans.items():
            if other != symbol:
                leaked |= np.asarray((test_index >= first) & (test_index <= last))
        total += len(test_index)
        covered += int(leaked.sum())
    return covered / total if total else 0.0


def _position_of(date):
    """Index position of `date` on the shared trading calendar."""
    return int(np.searchsorted(TRADING_DAYS, date))


# -- the dataset split: scripts/build_factor_datasets.py --------------------------


def test_cutoffs_are_ordered_and_derived_from_the_category_not_a_symbol():
    histories = _staggered_category()
    train_end, validation_end = category_cutoffs(histories)

    assert train_end < validation_end
    assert train_end > min(h.index.min() for h in histories.values())
    assert validation_end < max(h.index.max() for h in histories.values())
    # One pair of dates for the whole category is the entire point: category_cutoffs
    # takes the category, so there is no per-symbol variant that could drift.
    assert (train_end, validation_end) == category_cutoffs(histories)


def test_one_very_long_history_does_not_drag_the_cutoff_before_the_other_symbols():
    """^GSPC reaches back to 1927 while five of market_beta's seven symbols start after
    1998. Quantiles over the raw pooled calendar would put the train cutoff in that
    early era and leave the later symbols with no training rows at all; restricting to
    the median symbol's first date is what prevents it."""
    histories = _staggered_category()
    train_end, validation_end = category_cutoffs(histories)

    for symbol, history in histories.items():
        assert history.index.min() < train_end, f"{symbol} starts after the train cutoff"
        frames = split_by_cutoffs(history, train_end, validation_end)
        assert not frames["train"].empty, f"{symbol} got no training rows"


def test_rows_are_partitioned_at_the_cutoffs():
    history = _ohlcv(0, 2000, seed=11)
    train_end, validation_end = TRADING_DAYS[900], TRADING_DAYS[1400]
    frames = split_by_cutoffs(history, train_end, validation_end)

    assert frames["train"].index.max() < train_end
    assert frames["validation"].index.min() >= train_end
    assert frames["validation"].index.max() < validation_end
    assert frames["test"].index.min() >= validation_end

    # No row lands in two splits, and none is invented.
    combined = pd.concat([frames[name].index.to_series() for name in frames])
    assert combined.is_unique
    assert set(combined).issubset(set(history.index))


def test_embargo_covers_the_full_label_horizon():
    """Asserted against the label's own reach, not against EMBARGO_PERIODS: an embargo
    compared only to itself would still pass if someone set the constant to zero."""
    assert EMBARGO_PERIODS >= DEFAULT_LOOKFORWARD_PERIODS


def test_embargo_drops_lookforward_rows_from_train_and_validation_only():
    history = _ohlcv(0, 2000, seed=12)
    train_end, validation_end = TRADING_DAYS[900], TRADING_DAYS[1400]
    frames = split_by_cutoffs(history, train_end, validation_end)

    before_train_cutoff = history[history.index < train_end]
    in_validation_window = history[(history.index >= train_end) & (history.index < validation_end)]
    after_validation_cutoff = history[history.index >= validation_end]

    assert len(frames["train"]) <= len(before_train_cutoff) - DEFAULT_LOOKFORWARD_PERIODS
    assert len(frames["validation"]) <= len(in_validation_window) - DEFAULT_LOOKFORWARD_PERIODS
    assert len(frames["train"]) == len(before_train_cutoff) - EMBARGO_PERIODS
    assert len(frames["validation"]) == len(in_validation_window) - EMBARGO_PERIODS
    # Nothing follows test, so it has nothing to leak into and keeps every row.
    assert len(frames["test"]) == len(after_validation_cutoff)


def test_embargo_leaves_a_gap_no_label_window_can_span():
    """The reason the embargo exists: a swing label reads up to lookforward_periods
    bars ahead, so without a gap the last training rows are labelled from bars sitting
    on the validation side of the cutoff."""
    history = _ohlcv(0, 2000, seed=13)
    train_end, validation_end = TRADING_DAYS[900], TRADING_DAYS[1400]
    frames = split_by_cutoffs(history, train_end, validation_end)

    positions = {date: i for i, date in enumerate(history.index)}
    train_to_validation = (
        positions[frames["validation"].index.min()] - positions[frames["train"].index.max()]
    )
    validation_to_test = (
        positions[frames["test"].index.min()] - positions[frames["validation"].index.max()]
    )
    # Strictly greater than the label horizon: a row that many bars before the cutoff
    # is labelled from the last bar on its own side, not from one across the boundary.
    assert train_to_validation > DEFAULT_LOOKFORWARD_PERIODS
    assert validation_to_test > DEFAULT_LOOKFORWARD_PERIODS


@pytest.mark.parametrize("rows_in_window", [0, 1, EMBARGO_PERIODS - 1, EMBARGO_PERIODS])
def test_split_no_longer_than_the_embargo_comes_back_empty(rows_in_window):
    """A window with nothing left after the embargo must come back empty rather than
    wrapping around into a negative slice."""
    history = _ohlcv(0, 1000, seed=14)
    train_end = TRADING_DAYS[500]
    validation_end = history.index[_position_of(train_end) + rows_in_window]
    frames = split_by_cutoffs(history, train_end, validation_end)

    assert frames["validation"].empty
    assert not frames["train"].empty
    assert not frames["test"].empty


def test_symbol_starting_after_the_train_cutoff_gets_no_train_rows():
    """ARKK begins in late 2014, after growth_tech's train cutoff. It contributes
    validation and test rows and no training rows -- expected under a category-wide
    cutoff, and build_factor_datasets reports it rather than writing an unusable file."""
    histories = _staggered_category()
    train_end, validation_end = category_cutoffs(histories)
    start = _position_of(train_end) + 5
    latecomer = _ohlcv(start, 5800 - start, seed=21)  # runs to the same recent end

    frames = split_by_cutoffs(latecomer, train_end, validation_end)
    assert frames["train"].empty
    assert not frames["test"].empty


def test_no_test_row_falls_in_another_symbols_train_window():
    """The property the whole calendar split exists for, checked against the split it
    replaced so the assertion cannot pass vacuously."""
    histories = _staggered_category()
    train_end, validation_end = category_cutoffs(histories)

    calendar_aligned = {s: split_by_cutoffs(h, train_end, validation_end) for s, h in histories.items()}
    per_symbol = {s: _legacy_per_symbol_split(h) for s, h in histories.items()}

    assert _cross_symbol_leak_fraction(per_symbol) > 0.10, (
        "fixture no longer reproduces the leak the calendar split was written to fix"
    )
    assert _cross_symbol_leak_fraction(calendar_aligned) == 0.0


# -- the internal split: app/trainer.py -------------------------------------------


def _df_clean(symbol_lengths):
    """A prepare_training_data()-shaped frame: a `symbol` column, a shared date index,
    and one feature column holding each row's position in the frame.

    Symbols overlap on dates on purpose -- that overlap is what the label-based masking
    got wrong. Carrying the position as the feature value is what makes the returned
    X_train/X_val identifiable: the index alone can't distinguish AAA's 2003-04-01 row
    from BBB's.
    """
    frames = []
    for symbol, rows in symbol_lengths.items():
        frames.append(pd.DataFrame(
            {"symbol": symbol, "position": 0.0, "swing_label": 0},
            index=TRADING_DAYS[:rows],
        ))
    df_clean = pd.concat(frames)
    df_clean["position"] = np.arange(len(df_clean), dtype=float)
    return df_clean, df_clean[["position"]], df_clean["swing_label"]


def _selected_positions(subset):
    return set(subset["position"].astype(int).tolist())


def _symbol_positions(df_clean, symbol):
    return np.flatnonzero(df_clean["symbol"].to_numpy() == symbol)


def _run_internal_split(symbol_lengths, lookforward_periods=10):
    trainer = SwingTradeTrainer(lookforward_periods=lookforward_periods)
    df_clean, X, y = _df_clean(symbol_lengths)
    slices = trainer._chronological_internal_split(df_clean, X, y)
    selected = {name: _selected_positions(features) for name, (features, _) in slices.items()}
    for name, (features, labels) in slices.items():
        assert len(features) == len(labels), f"{name} slice has mismatched X/y lengths"
    return trainer, df_clean, selected


def test_internal_split_returns_three_disjoint_slices():
    _, df_clean, selected = _run_internal_split({"AAA": 1000, "BBB": 600})
    assert set(selected) == {"fit", "calibrate", "report"}
    assert selected["fit"].isdisjoint(selected["calibrate"])
    assert selected["fit"].isdisjoint(selected["report"])
    assert selected["calibrate"].isdisjoint(selected["report"])
    assert all(slice_positions for slice_positions in selected.values()), "no slice may be empty here"


def test_internal_split_orders_slices_chronologically_per_symbol():
    """fit before calibrate before report, within each symbol. Calibrating on rows that
    precede the fitting rows would be the same look-ahead in a different costume."""
    lengths = {"AAA": 1000, "BBB": 600}
    _, df_clean, selected = _run_internal_split(lengths)
    for symbol in lengths:
        owned = set(_symbol_positions(df_clean, symbol).tolist())
        fit = selected["fit"] & owned
        calibrate = selected["calibrate"] & owned
        report = selected["report"] & owned
        assert max(fit) < min(calibrate)
        assert max(calibrate) < min(report)


def test_internal_split_embargoes_both_boundaries():
    lookforward = 10
    lengths = {"AAA": 1000, "BBB": 600}
    trainer, df_clean, selected = _run_internal_split(lengths, lookforward)
    for symbol in lengths:
        owned = set(_symbol_positions(df_clean, symbol).tolist())
        assert min(selected["calibrate"] & owned) - max(selected["fit"] & owned) - 1 == lookforward
        assert min(selected["report"] & owned) - max(selected["calibrate"] & owned) - 1 == lookforward
    # Two boundaries per symbol, lookforward rows each.
    assert trainer._embargoed_rows == 2 * lookforward * len(lengths)


def test_internal_split_accounts_for_every_row():
    lengths = {"AAA": 1000, "BBB": 600, "CCC": 400}
    trainer, df_clean, selected = _run_internal_split(lengths)
    assigned = sum(len(positions) for positions in selected.values())
    assert assigned + trainer._embargoed_rows == len(df_clean)


def test_internal_split_holds_out_the_most_recent_slice_of_each_symbol():
    lengths = {"AAA": 1000, "BBB": 600}
    _, df_clean, selected = _run_internal_split(lengths)
    for symbol, rows in lengths.items():
        positions = _symbol_positions(df_clean, symbol)
        report_start = int(rows * (1 - INTERNAL_VALIDATION_FRACTION))
        assert selected["report"] & set(positions.tolist()) == set(positions[report_start:].tolist())


def test_calibration_slice_is_the_configured_share():
    lengths = {"AAA": 1000}
    lookforward = 10
    _, df_clean, selected = _run_internal_split(lengths, lookforward)
    positions = _symbol_positions(df_clean, "AAA")
    expected_start = int(1000 * (1 - INTERNAL_VALIDATION_FRACTION - INTERNAL_CALIBRATION_FRACTION))
    expected_end = int(1000 * (1 - INTERNAL_VALIDATION_FRACTION)) - lookforward
    assert selected["calibrate"] == set(positions[expected_start:expected_end].tolist())


def test_internal_split_is_positional_not_label_based():
    """Regression test. Symbols share dates, and the split used to build its mask with
    `mask.loc[symbol_slice.index] = False`. With duplicate index labels that assignment
    also flips every other symbol's rows on those dates, so holding out short-history
    BBB's recent rows silently held out long-history AAA's rows on the same dates.

    AAA has 1000 rows and BBB the first 500 of the same dates, so BBB's held-out tail
    sits in the middle of AAA's fitting range.
    """
    lookforward = 10
    lengths = {"AAA": 1000, "BBB": 500}
    _, df_clean, selected = _run_internal_split(lengths, lookforward)

    aaa = _symbol_positions(df_clean, "AAA")
    bbb = _symbol_positions(df_clean, "BBB")
    expected_fit_end = int(1000 * (1 - INTERNAL_VALIDATION_FRACTION - INTERNAL_CALIBRATION_FRACTION)) - lookforward
    assert len(selected["fit"] & set(aaa.tolist())) == expected_fit_end

    bbb_holdout_start = int(len(bbb) * (1 - INTERNAL_VALIDATION_FRACTION))
    contested_dates = df_clean.index[bbb[bbb_holdout_start:]]
    contested_aaa = aaa[np.isin(df_clean.index[aaa], contested_dates)]
    assert len(contested_aaa) > 0, "fixture no longer overlaps the two symbols' dates"
    assert set(contested_aaa.tolist()).issubset(selected["fit"]), (
        "AAA rows were held out because BBB shares those dates -- the label-based bug"
    )


def test_symbol_too_short_for_a_calibration_slice_contributes_none():
    """A two-row calibration set produces a nonsense probability mapping, so a symbol
    that cannot leave a usable one contributes to fit and report only."""
    lengths = {"AAA": 1000, "TINY": 12}
    _, df_clean, selected = _run_internal_split(lengths, lookforward_periods=10)
    tiny = set(_symbol_positions(df_clean, "TINY").tolist())
    assert not (selected["calibrate"] & tiny)
    assert selected["report"] & tiny
