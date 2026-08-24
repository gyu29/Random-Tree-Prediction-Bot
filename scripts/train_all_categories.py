"""Trains and evaluates one model per factor category.

For each category: trains on train/<category>/*.csv, saves a signed/versioned
model via app.model_registry, then evaluates on test/<category>/*.csv -- which
is, by construction (see scripts/build_factor_datasets.py), strictly
chronologically after that symbol's training data. That's the actual
walk-forward, out-of-sample check; the "internal validation" score printed
during training (see app/trainer.py) is secondary.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import model_registry  # noqa: E402
from app.config import (  # noqa: E402
    CALIBRATED_DECISION_THRESHOLDS,
    CALIBRATED_SWING_THRESHOLDS,
    DEFAULT_SWING_THRESHOLD,
)
from app.data_loader import DataProcessor, category_test_dir, list_categories  # noqa: E402
from app.detector import NoModelAvailableError, SwingTradeDetector, walk_forward_backtest  # noqa: E402
from app.market_data.alpha_vantage_provider import AlphaVantageProvider  # noqa: E402
from app.trading_system import SwingTradingSystem  # noqa: E402


def evaluate_on_test_split(category, detector):
    """Runs the walk-forward backtest against every symbol's held-out test CSV,
    concatenating trades so one badly-behaved symbol doesn't hide the rest."""
    test_dir = category_test_dir(category)
    symbol_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".csv")) if os.path.isdir(test_dir) else []

    all_trades = []
    per_symbol = []
    for filename in symbol_files:
        symbol = DataProcessor.infer_symbol_from_path(filename).upper()
        path = os.path.join(test_dir, filename)
        try:
            df = DataProcessor.load_and_validate_data(path)
            result = walk_forward_backtest(detector, df)
        except Exception as error:
            per_symbol.append({"symbol": symbol, "error": str(error)})
            continue
        for trade in result["trades"]:
            trade["symbol"] = symbol
        all_trades.extend(result["trades"])
        per_symbol.append({
            "symbol": symbol, "num_trades": result["num_trades"], "win_rate": result["win_rate"],
            "total_return": result["total_return"], "sharpe": result["sharpe"],
            "max_drawdown": result["max_drawdown"], "peak_probability": result["peak_probability"],
        })

    profits = [t["profit_pct"] for t in all_trades]
    combined = {
        "num_trades": len(all_trades),
        "win_rate": (len([p for p in profits if p > 0]) / len(profits)) if profits else 0.0,
        "avg_profit": float(np.mean(profits)) if profits else 0.0,
        "total_return_compounded": float(np.prod([1 + p for p in profits]) - 1) if profits else 0.0,
    }
    return combined, per_symbol


def train_and_evaluate_category(system, category):
    print(f"\n{'=' * 70}\n{category}\n{'=' * 70}")
    swing_threshold = CALIBRATED_SWING_THRESHOLDS.get(category, DEFAULT_SWING_THRESHOLD)
    start = time.time()
    train_result = system.train_model(category, swing_threshold=swing_threshold)
    elapsed = time.time() - start
    stats = train_result["training_stats"]
    print(f"Trained in {elapsed:.1f}s | swing_threshold={swing_threshold:.0%} "
          f"| validation_score={train_result['validation_score']:.4f} "
          f"| scale_pos_weight={stats['scale_pos_weight']:.2f} "
          f"| effective_swing_threshold={stats['effective_swing_threshold']:.2%}")

    if category in CALIBRATED_DECISION_THRESHOLDS:
        decision_threshold = CALIBRATED_DECISION_THRESHOLDS[category]
        model_registry.update_decision_threshold(category, decision_threshold)
        print(f"Applied calibrated decision_threshold={decision_threshold:.0%} (see MODEL_CALIBRATION_FINDINGS.md)")

    # Fresh detector loaded straight from the signed manifest, exactly like a real user's
    # session would -- this also exercises the integrity check end to end.
    provider = AlphaVantageProvider(api_key=None)
    detector = SwingTradeDetector(category, provider)
    if not detector.is_ready:
        raise NoModelAvailableError(f"Just-trained category '{category}' failed to reload")

    combined, per_symbol = evaluate_on_test_split(category, detector)
    print(f"Out-of-sample test-split backtest: {combined['num_trades']} trades, "
          f"win_rate={combined['win_rate']:.1%}, avg_profit={combined['avg_profit']:.2%}, "
          f"compounded_return={combined['total_return_compounded']:.2%}")
    for row in per_symbol:
        if "error" in row:
            print(f"  {row['symbol']}: skipped ({row['error']})")
        else:
            print(f"  {row['symbol']}: {row['num_trades']} trades, win_rate={row['win_rate']:.1%}, "
                  f"return={row['total_return']:.2%}, peak_probability={row['peak_probability']:.1%}")

    return {
        "category": category,
        "swing_threshold": swing_threshold,
        "decision_threshold": detector.decision_threshold,
        "validation_score": train_result["validation_score"],
        "scale_pos_weight": stats["scale_pos_weight"],
        "test_trades": combined["num_trades"],
        "test_win_rate": combined["win_rate"],
        "test_avg_profit": combined["avg_profit"],
        "test_compounded_return": combined["total_return_compounded"],
    }


def main():
    system = SwingTradingSystem()
    summary_rows = []
    failures = []
    for category in list_categories():
        try:
            summary_rows.append(train_and_evaluate_category(system, category))
        except Exception as error:
            print(f"FAILED to train/evaluate '{category}': {error}")
            failures.append((category, str(error)))

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print(f"\n{len(summary_rows)}/{len(list_categories())} categories trained successfully.")
    if failures:
        print("Failures:")
        for category, error in failures:
            print(f"  {category}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
