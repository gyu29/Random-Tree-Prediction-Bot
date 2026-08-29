"""Downloads each factor category's tickers and writes the train/validation/test split.

The split is chronological *and calendar-aligned across the whole category*: one pair
of cutoff dates per category, applied to every symbol in it.

The earlier per-symbol percentage split (oldest 55% of each symbol's own rows to
train, and so on) was honest for a single ticker and porous for the pooled category,
because the symbols inside a category are near-duplicates of each other. market_beta
holds SPY, VOO, IVV, VTI, DIA, ^DJI and ^GSPC -- seven wrappers around the same US
large-cap tape, with a median pairwise daily-return correlation of 0.98. Under the old
split ^GSPC's test rows covered 1997-2026 while SPY (1993-2011), DIA (1998-2013), IVV
(2000-2014) and VOO (2010-2019) were still in *train*, so 39% of market_beta's test
rows fell on calendar days the model had already trained on through a sibling ticker.
For energy_commodity it was 49%, for growth_tech 42%. Those categories scored better
on test than on validation, which is the signature of the leak rather than of skill.

One cutoff per category closes that: after this, no symbol's test rows share a date
with any symbol's training rows. Expect the reported numbers to drop -- that is the
fix working, not a regression.

Two consequences worth knowing about:

  * Symbols with short histories contribute fewer rows, and one starting after the
    train cutoff contributes no training rows at all (ARKK, which begins in late 2014,
    is the likely case). That symbol is still written for the splits it does cover;
    the run warns rather than silently shipping a near-empty training file.
  * The cutoffs are derived from the data, so they move as history grows. Pin them in
    app.config.CALENDAR_SPLIT_CUTOFFS when a run needs to be reproducible.

Both split seams also carry an embargo: the last EMBARGO_PERIODS rows of train and of
validation are dropped. A swing label looks that many bars into the future, so those
rows' labels are decided by bars sitting on the far side of the cutoff.
"""
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.market_context import fetch_context_series, save_context  # noqa: E402
from app.config import (  # noqa: E402
    CALENDAR_SPLIT_CUTOFFS,
    DEFAULT_LOOKFORWARD_PERIODS,
    FACTOR_CATEGORIES,
    TEST_ROOT,
    TRAIN_ROOT,
    VALIDATION_ROOT,
)

TRAIN_FRACTION = 0.55
VALIDATION_FRACTION = 0.15
# test gets the remaining 0.30 -- unchanged from the original 70/30 split's test share.
# The fractions now pick *dates* (the quantiles of the category's pooled row calendar)
# rather than slicing each symbol's own rows, so a category's realized split will land
# near these proportions without matching them exactly.

# Rows dropped at each end-of-split boundary. Matches DEFAULT_LOOKFORWARD_PERIODS
# because that is how far app/labeling.py's swing label reaches forward.
EMBARGO_PERIODS = DEFAULT_LOOKFORWARD_PERIODS

# A split below this many rows is not usable for its purpose (training needs enough
# rows to fit on; validation/test need enough to measure anything). Written anyway if
# non-empty, but reported so the run doesn't look clean when it isn't.
MIN_USABLE_SPLIT_ROWS = 250

SPLIT_ROOTS = {"train": TRAIN_ROOT, "validation": VALIDATION_ROOT, "test": TEST_ROOT}


def _safe_filename(ticker):
    return ticker.replace("^", "").replace("/", "_").replace("\\", "_").strip()


def category_cutoffs(histories):
    """The two calendar cutoffs for one category, as (train_end, validation_end).

    Derived from the quantiles of the category's pooled trading days, restricted to
    the window where the category actually has broad coverage: the median symbol's
    first date. Without that restriction one long series dominates -- ^GSPC's history
    reaches back to 1927, and pooling it raw would put market_beta's train cutoff in
    the early 1980s, before five of its seven symbols exist.

    Rows earlier than that common start are not discarded; they sit before the train
    cutoff either way and stay in train as extra history.
    """
    first_dates = sorted(history.index.min() for history in histories.values())
    common_start = first_dates[len(first_dates) // 2]

    pooled = pd.DatetimeIndex(
        [date for history in histories.values() for date in history.index if date >= common_start]
    ).sort_values()
    if pooled.empty:
        raise ValueError("no rows at or after the category's common start date")

    train_end = pooled[int(len(pooled) * TRAIN_FRACTION)]
    validation_end = pooled[int(len(pooled) * (TRAIN_FRACTION + VALIDATION_FRACTION))]
    return train_end, validation_end


def split_by_cutoffs(df, train_end, validation_end):
    """Slices one symbol at the category's cutoffs, embargoing each seam.

    Train and validation each lose their final EMBARGO_PERIODS rows; test keeps all of
    its (nothing follows it to leak into).
    """
    df = df.sort_index()
    train_df = df[df.index < train_end]
    validation_df = df[(df.index >= train_end) & (df.index < validation_end)]
    test_df = df[df.index >= validation_end]

    if EMBARGO_PERIODS:
        train_df = train_df.iloc[:-EMBARGO_PERIODS] if len(train_df) > EMBARGO_PERIODS else train_df.iloc[:0]
        validation_df = (
            validation_df.iloc[:-EMBARGO_PERIODS]
            if len(validation_df) > EMBARGO_PERIODS
            else validation_df.iloc[:0]
        )
    return {"train": train_df, "validation": validation_df, "test": test_df}


def _write_splits(frames, category, ticker):
    """Writes each non-empty split. An empty frame is skipped rather than written as a
    header-only CSV, which app/data_loader.py would only reject later at load time."""
    safe_name = _safe_filename(ticker)
    written = {}
    for split_name, frame in frames.items():
        split_dir = os.path.join(SPLIT_ROOTS[split_name], category)
        os.makedirs(split_dir, exist_ok=True)
        path = os.path.join(split_dir, f"{safe_name}.csv")
        if frame.empty:
            if os.path.exists(path):  # stale file from an earlier, differently-split run
                os.remove(path)
            continue
        frame.to_csv(path)
        written[split_name] = frame
    return written


def prune_removed_symbols(category, tickers):
    """Deletes split CSVs for symbols no longer in the category.

    Without this, changing FACTOR_CATEGORIES leaves the old files on disk and every
    downstream step keeps training on them: dropping SPY, VOO, IVV and VTI from
    market_beta did not remove them, so the next run would have quietly restored the
    seven-clones-of-one-index problem the change existed to fix -- and kept training on
    SPY, which app/market_context.py uses as the benchmark and whose excess-return
    columns are therefore identically zero.
    """
    keep = {_safe_filename(ticker) for ticker in tickers}
    removed = []
    for split_name, root in SPLIT_ROOTS.items():
        directory = os.path.join(root, category)
        if not os.path.isdir(directory):
            continue
        for filename in sorted(f for f in os.listdir(directory) if f.endswith(".csv")):
            if filename[:-4] not in keep:
                os.remove(os.path.join(directory, filename))
                removed.append(f"{split_name}/{filename[:-4]}")
    return removed


def _describe(split_name, frame):
    if frame.empty:
        return f"{split_name}: 0 rows"
    return (
        f"{split_name}: {len(frame)} rows "
        f"({frame.index.min().date()} -> {frame.index.max().date()})"
    )


def _download(ticker):
    history = yf.Ticker(ticker).history(period="max")
    if "Capital Gains" in history.columns:
        history = history.drop(columns=["Capital Gains"])
    if history.empty or len(history) < 100:
        raise ValueError(f"insufficient history ({len(history)} rows)")
    return history.sort_index()


def build_factor_datasets(categories=None):
    categories = categories or FACTOR_CATEGORIES
    print("Fetching shared market-context series (app/market_context.py)...")
    try:
        path = save_context(fetch_context_series())
        print(f"  wrote {path}")
    except Exception as error:
        print(f"  FAILED to build market context: {error}. Models will train without it.")
    summary_rows = []
    failures = []
    warnings = []

    for category, tickers in categories.items():
        print(f"\n=== {category} ({len(tickers)} tickers) ===")

        histories = {}
        for ticker in tickers:
            try:
                print(f"Downloading {ticker}...")
                histories[ticker] = _download(ticker)
            except Exception as error:
                print(f"  FAILED {ticker}: {error}")
                failures.append((category, ticker, str(error)))
        if not histories:
            continue

        try:
            if category in CALENDAR_SPLIT_CUTOFFS:
                configured = CALENDAR_SPLIT_CUTOFFS[category]
                sample_index = next(iter(histories.values())).index
                train_end, validation_end = (
                    pd.Timestamp(configured[0], tz=sample_index.tz),
                    pd.Timestamp(configured[1], tz=sample_index.tz),
                )
                source = "configured"
            else:
                train_end, validation_end = category_cutoffs(histories)
                source = "derived"
        except Exception as error:
            print(f"  FAILED to pick cutoffs for {category}: {error}")
            failures.append((category, "<cutoffs>", str(error)))
            continue

        print(f"  Cutoffs ({source}): train < {train_end.date()} "
              f"<= validation < {validation_end.date()} <= test  "
              f"[{EMBARGO_PERIODS}-row embargo at each seam]")

        pruned = prune_removed_symbols(category, tickers)
        if pruned:
            print(f"  Removed {len(pruned)} file(s) for symbols no longer in this category: "
                  f"{', '.join(sorted({name.split('/')[1] for name in pruned}))}")

        for ticker, history in histories.items():
            frames = split_by_cutoffs(history, train_end, validation_end)
            written = _write_splits(frames, category, ticker)

            row = {"category": category, "ticker": ticker,
                   "train_cutoff": train_end.date(), "validation_cutoff": validation_end.date()}
            for split_name, frame in frames.items():
                row[f"{split_name}_rows"] = len(frame)
                row[f"{split_name}_start"] = frame.index.min() if not frame.empty else None
                row[f"{split_name}_end"] = frame.index.max() if not frame.empty else None
            summary_rows.append(row)

            print(f"  {ticker}: " + "  ".join(_describe(name, frames[name]) for name in SPLIT_ROOTS))
            for split_name, frame in frames.items():
                if len(frame) < MIN_USABLE_SPLIT_ROWS:
                    detail = (f"{category}/{ticker}: {split_name} has {len(frame)} rows "
                              f"(< {MIN_USABLE_SPLIT_ROWS}); history starts "
                              f"{history.index.min().date()}, after the cutoff for this split")
                    warnings.append(detail)
                    if split_name not in written:
                        print(f"    NOTE: no {split_name} file written -- this symbol has no rows in that window")

    print("\n=== Summary ===")
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print(f"\n{len(summary_rows)} succeeded, {len(failures)} failed.")

    if warnings:
        print(f"\n{len(warnings)} thin split(s) -- expected where a symbol's history starts "
              f"after its category's cutoff, not an error:")
        for detail in warnings:
            print(f"  {detail}")
    if failures:
        print("Failures:")
        for category, ticker, error in failures:
            print(f"  {category}/{ticker}: {error}")

    return summary_df, failures


if __name__ == "__main__":
    _, build_failures = build_factor_datasets()
    sys.exit(1 if build_failures else 0)
