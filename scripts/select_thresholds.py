"""Selects decision_threshold per category using validation/ only, then confirms the
choice exactly once against test/ -- the two-step process the earlier test-set threshold
sweep skipped (see app/trainer.py's module docstring, and the caveats in the walk-forward
backtest report from this same session).

validation/ is picked over: it's never used to fit the model (only train/ is) and it's
chronologically before test/, so searching over it doesn't touch the one slice meant to
answer "how would this actually perform going forward."
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data_loader import DataProcessor, category_test_dir, category_validation_dir, list_categories  # noqa: E402
from app.detector import SwingTradeDetector, walk_forward_backtest  # noqa: E402
from app.market_data.alpha_vantage_provider import AlphaVantageProvider  # noqa: E402

THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80]
MIN_TRADES_FOR_SELECTION = 5  # below this, fall back to the default rather than trust the search


def load_symbol_data(directory):
    data = {}
    if not os.path.isdir(directory):
        return data
    for filename in sorted(f for f in os.listdir(directory) if f.endswith(".csv")):
        symbol = DataProcessor.infer_symbol_from_path(filename).upper()
        try:
            data[symbol] = DataProcessor.load_and_validate_data(os.path.join(directory, filename))
        except Exception:
            continue
    return data


def pooled_metrics(detector, symbol_data, threshold):
    all_profits = []
    drawdowns = []
    for df in symbol_data.values():
        try:
            result = walk_forward_backtest(detector, df, decision_threshold=threshold)
        except ValueError:
            continue  # not enough rows for this symbol's window at this category's lookforward
        all_profits.extend(trade["profit_pct"] for trade in result["trades"])
        drawdowns.append(result["max_drawdown"])

    returns = np.asarray(all_profits) if all_profits else np.asarray([0.0])
    sharpe = float(np.mean(returns) / np.std(returns) * (12 ** 0.5)) if np.std(returns) else 0.0
    return {
        "num_trades": len(all_profits),
        "win_rate": (len([p for p in all_profits if p > 0]) / len(all_profits)) if all_profits else 0.0,
        "compounded_return": float(np.prod([1 + p for p in all_profits]) - 1) if all_profits else 0.0,
        "sharpe": sharpe,
        "avg_max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
    }


def select_for_category(category):
    provider = AlphaVantageProvider(api_key=None)
    detector = SwingTradeDetector(category, provider)
    default_threshold = detector.decision_threshold

    validation_data = load_symbol_data(category_validation_dir(category))
    test_data = load_symbol_data(category_test_dir(category))

    validation_sweep = [
        {"threshold": t, **pooled_metrics(detector, validation_data, t)}
        for t in THRESHOLDS
    ]
    reliable = [r for r in validation_sweep if r["num_trades"] >= MIN_TRADES_FOR_SELECTION]
    if reliable:
        selected = max(reliable, key=lambda r: r["sharpe"])["threshold"]
        selection_note = f"best validation Sharpe among thresholds with >={MIN_TRADES_FOR_SELECTION} trades"
    else:
        selected = default_threshold
        selection_note = f"no threshold reached {MIN_TRADES_FOR_SELECTION} validation trades -- kept default"

    selected_on_validation = pooled_metrics(detector, validation_data, selected)
    selected_on_test = pooled_metrics(detector, test_data, selected)
    default_on_test = pooled_metrics(detector, test_data, default_threshold)

    return {
        "category": category,
        "default_threshold": default_threshold,
        "selected_threshold": selected,
        "selection_note": selection_note,
        "validation_symbols": len(validation_data),
        "test_symbols": len(test_data),
        "validation_sweep": validation_sweep,
        "selected_on_validation": selected_on_validation,
        "selected_on_test": selected_on_test,
        "default_on_test": default_on_test,
    }


def main():
    rows = []
    for category in list_categories():
        print(f"\n{'=' * 70}\n{category}\n{'=' * 70}")
        result = select_for_category(category)
        print(f"Selected threshold: {result['selected_threshold']:.0%} ({result['selection_note']})")
        print(f"  validation @ selected: {result['selected_on_validation']}")
        print(f"  test @ selected:       {result['selected_on_test']}")
        print(f"  test @ default 65%:    {result['default_on_test']}")
        rows.append(result)

    print(f"\n{'=' * 70}\nSummary: selected threshold, validation perf, and the honest test-set check\n{'=' * 70}")
    summary = pd.DataFrame([{
        "category": r["category"],
        "selected_thr": r["selected_threshold"],
        "val_trades": r["selected_on_validation"]["num_trades"],
        "val_sharpe": r["selected_on_validation"]["sharpe"],
        "val_return": r["selected_on_validation"]["compounded_return"],
        "test_trades_selected": r["selected_on_test"]["num_trades"],
        "test_sharpe_selected": r["selected_on_test"]["sharpe"],
        "test_return_selected": r["selected_on_test"]["compounded_return"],
        "test_return_default65": r["default_on_test"]["compounded_return"],
    } for r in rows])
    print(summary.to_string(index=False))

    out_path = os.environ.get("THRESHOLD_SELECTION_OUTPUT")
    if out_path:
        import json
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\nFull sweep detail written to {out_path}")


if __name__ == "__main__":
    main()
