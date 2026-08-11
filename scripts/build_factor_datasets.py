"""Builds the train/, validation/, and test/ factor-model datasets from Yahoo Finance.

Replaces download_sp500_data.py. For every symbol in FACTOR_CATEGORIES, downloads full
history and splits it *per symbol, chronologically* into three pieces: the oldest
TRAIN_FRACTION goes to train/<category>/<symbol>.csv (the only data models are ever
fit on), the next VALIDATION_FRACTION goes to validation/<category>/<symbol>.csv (for
choosing things like decision_threshold -- see scripts/select_thresholds.py -- without
touching test), and the newest remainder goes to test/<category>/<symbol>.csv (touched
exactly once, to report final performance of whatever was chosen using train+validation
alone). Because the split is chronological per symbol, train < validation < test in
time for every symbol -- a walk-forward split, not a random one.
"""
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import FACTOR_CATEGORIES, TRAIN_ROOT, VALIDATION_ROOT, TEST_ROOT  # noqa: E402

TRAIN_FRACTION = 0.55
VALIDATION_FRACTION = 0.15
# test gets the remaining 0.30 -- unchanged from the original 70/30 split's test share,
# so the train+validation/test boundary sits exactly where the old train/test boundary did.


def _safe_filename(ticker):
    return ticker.replace("^", "").replace("/", "_").replace("\\", "_").strip()


def _split_and_write(df, category, ticker):
    df = df.sort_index()
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_FRACTION)
    validation_end = int(n_rows * (TRAIN_FRACTION + VALIDATION_FRACTION))
    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:]

    dirs = {"train": TRAIN_ROOT, "validation": VALIDATION_ROOT, "test": TEST_ROOT}
    frames = {"train": train_df, "validation": validation_df, "test": test_df}
    safe_name = _safe_filename(ticker)
    for split_name, root in dirs.items():
        split_dir = os.path.join(root, category)
        os.makedirs(split_dir, exist_ok=True)
        frames[split_name].to_csv(os.path.join(split_dir, f"{safe_name}.csv"))

    return train_df, validation_df, test_df


def build_factor_datasets(categories=None):
    categories = categories or FACTOR_CATEGORIES
    summary_rows = []
    failures = []

    for category, tickers in categories.items():
        print(f"\n=== {category} ({len(tickers)} tickers) ===")
        for ticker in tickers:
            try:
                print(f"Downloading {ticker}...")
                history = yf.Ticker(ticker).history(period="max")
                if "Capital Gains" in history.columns:
                    history = history.drop(columns=["Capital Gains"])
                if history.empty or len(history) < 100:
                    raise ValueError(f"insufficient history ({len(history)} rows)")

                train_df, validation_df, test_df = _split_and_write(history, category, ticker)
                summary_rows.append({
                    "category": category,
                    "ticker": ticker,
                    "train_rows": len(train_df),
                    "train_start": train_df.index.min(),
                    "train_end": train_df.index.max(),
                    "validation_rows": len(validation_df),
                    "validation_start": validation_df.index.min(),
                    "validation_end": validation_df.index.max(),
                    "test_rows": len(test_df),
                    "test_start": test_df.index.min(),
                    "test_end": test_df.index.max(),
                })
                print(
                    f"  train: {len(train_df)} rows ({train_df.index.min().date()} -> {train_df.index.max().date()})  "
                    f"validation: {len(validation_df)} rows ({validation_df.index.min().date()} -> {validation_df.index.max().date()})  "
                    f"test: {len(test_df)} rows ({test_df.index.min().date()} -> {test_df.index.max().date()})"
                )
            except Exception as error:
                print(f"  FAILED {ticker}: {error}")
                failures.append((category, ticker, str(error)))

    print("\n=== Summary ===")
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print(f"\n{len(summary_rows)} succeeded, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for category, ticker, error in failures:
            print(f"  {category}/{ticker}: {error}")

    return summary_df, failures


if __name__ == "__main__":
    _, build_failures = build_factor_datasets()
    sys.exit(1 if build_failures else 0)
