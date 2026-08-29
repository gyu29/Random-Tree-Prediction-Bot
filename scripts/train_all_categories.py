"""Trains and evaluates one model per factor category.

For each category: trains on train/<category>/*.csv, saves a signed/versioned
model via app.model_registry, then evaluates on test/<category>/*.csv -- which
is, by construction (see scripts/build_factor_datasets.py), strictly
chronologically after every symbol's training data in that category. That's the
actual walk-forward, out-of-sample check; the "internal validation" score printed
during training (see app/trainer.py) is secondary.

The test-split evaluation has two halves, and both are printed:

  * Classification metrics -- PR-AUC, ROC-AUC, precision, recall, and accuracy
    shown next to the always-negative baseline. Accuracy alone is not reportable
    here: positive labels run 0.6-8% of rows, so predicting "no swing" on every
    row scores 92-99% and beats anything this trainer produces. PR-AUC against
    the base rate is the number that says whether the model ranks better than
    chance.
  * The walk-forward backtest -- trade count, win rate, realized return. A model
    can rank well and still trade badly (and the reverse), so neither half
    substitutes for the other.

The backtest is run twice: once at the category's decision_threshold and once with
the threshold at zero, which drives the same entry/exit machinery with the model's
ranking ignored. The difference is the only number that answers "is this model worth
consulting at all", and it is reported in standard errors because on 30-200 trades a
raw difference in mean return is mostly noise. A category that cannot beat its own
null belongs in app.config.CATEGORIES_FAILING_VALIDATION, which is what gates it out
of alerts and the screener ranking.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import model_registry  # noqa: E402
from app.config import (  # noqa: E402
    CALIBRATED_DECISION_THRESHOLDS,
    CALIBRATED_SWING_THRESHOLDS,
    DEFAULT_SWING_THRESHOLD,
)
from app.data_loader import DataProcessor, category_test_dir, list_categories  # noqa: E402
from app.detector import (  # noqa: E402
    NoModelAvailableError,
    SwingTradeDetector,
    resolve_market_context,
    walk_forward_backtest,
)
from app.indicators import TechnicalIndicators  # noqa: E402
from app.labeling import create_swing_labels  # noqa: E402
from app.market_data.alpha_vantage_provider import AlphaVantageProvider  # noqa: E402
from app.trading_system import SwingTradingSystem  # noqa: E402


def classification_metrics_on_test_split(category, detector):
    """Scores every test row with the detector's own model and labels, and returns the
    metrics that survive a heavily imbalanced target.

    Labels are rebuilt with the same swing_threshold/lookforward/min_hold the model was
    trained under (read back off the signed manifest via the detector), so the target
    here is exactly the one the model was fit to predict.
    """
    test_dir = category_test_dir(category)
    symbol_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".csv")) if os.path.isdir(test_dir) else []

    frames = []
    for filename in symbol_files:
        path = os.path.join(test_dir, filename)
        try:
            df = DataProcessor.load_and_validate_data(path)
            features = TechnicalIndicators.create_all_indicators(
                df.drop(columns=["symbol"], errors="ignore"),
                market_context=resolve_market_context(detector),
            )
            labeled_frame = create_swing_labels(
                features, detector.swing_threshold, detector.lookforward_periods,
                detector.training_stats.get("min_hold_periods", 3),
            )
        # Filled per symbol and forward only, before the concat -- the same discipline as
        # app/trainer.py. A frame-wide fill lets one symbol supply another's missing
        # columns, and a back-fill writes later values into earlier rows.
            fill_cols = [c for c in labeled_frame.columns if c != "swing_label"]
            labeled_frame[fill_cols] = labeled_frame[fill_cols].ffill()
            frames.append(labeled_frame)
        except Exception:
            continue  # the backtest half reports this symbol's error already
    if not frames:
        return None

    labeled = pd.concat(frames)
    for feature in detector.feature_columns:
        if feature not in labeled.columns:
            labeled[feature] = np.nan
    X = labeled[detector.feature_columns].replace([np.inf, -np.inf], 0).fillna(0)
    y = labeled["swing_label"].astype(int).to_numpy()

    probabilities = detector.model.predict_proba(detector.scaler.transform(X))[:, 1]
    predictions = (probabilities >= detector.decision_threshold).astype(int)

    base_rate = float(y.mean()) if len(y) else 0.0
    has_both_classes = 0 < base_rate < 1
    return {
        "rows": len(y),
        "base_rate": base_rate,
        "accuracy": float(accuracy_score(y, predictions)),
        "majority_accuracy": 1.0 - base_rate,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probabilities)) if has_both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, probabilities)) if has_both_classes else float("nan"),
        "signals": int(predictions.sum()),
    }


def evaluate_on_test_split(category, detector, decision_threshold=None):
    """Runs the walk-forward backtest against every symbol's held-out test CSV,
    concatenating trades so one badly-behaved symbol doesn't hide the rest.

    decision_threshold=None uses the detector's own; pass 0.0 for the model-off null.
    """
    test_dir = category_test_dir(category)
    symbol_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".csv")) if os.path.isdir(test_dir) else []

    all_trades = []
    per_symbol = []
    for filename in symbol_files:
        symbol = DataProcessor.infer_symbol_from_path(filename).upper()
        path = os.path.join(test_dir, filename)
        try:
            df = DataProcessor.load_and_validate_data(path)
            result = walk_forward_backtest(detector, df, decision_threshold=decision_threshold)
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
        "std_profit": float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0,
        "total_return_compounded": float(np.prod([1 + p for p in profits]) - 1) if profits else 0.0,
    }
    return combined, per_symbol


def compare_against_null(combined, null_combined):
    """Is the model's edge over "ignore the model" bigger than its own noise?

    Reported in standard errors of the model's own mean: below 1 the difference is
    indistinguishable from sampling noise, and negative means the ranking actively hurt.
    """
    trades = combined["num_trades"]
    standard_error = combined["std_profit"] / np.sqrt(trades) if trades > 1 else float("inf")
    edge = combined["avg_profit"] - null_combined["avg_profit"]
    ratio = edge / standard_error if standard_error and np.isfinite(standard_error) else 0.0
    if ratio > 2:
        verdict = "real edge over the null"
    elif ratio > 1:
        verdict = "suggestive, not established"
    elif ratio > -1:
        verdict = "indistinguishable from ignoring the model"
    else:
        verdict = "WORSE than ignoring the model -- gate this category"
    return {"edge": edge, "standard_error": standard_error, "edge_in_standard_errors": ratio,
            "verdict": verdict}


def train_and_evaluate_category(system, category):
    print(f"\n{'=' * 70}\n{category}\n{'=' * 70}")
    swing_threshold = CALIBRATED_SWING_THRESHOLDS.get(category, DEFAULT_SWING_THRESHOLD)
    start = time.time()
    train_result = system.train_model(category, swing_threshold=swing_threshold)
    elapsed = time.time() - start
    stats = train_result["training_stats"]
    print(f"Trained in {elapsed:.1f}s | swing_threshold={swing_threshold:.0%} "
          f"| internal validation PR-AUC={stats['validation_pr_auc']:.4f} "
          f"(base rate {stats['validation_base_rate']:.2%}) "
          f"| scale_pos_weight={stats['scale_pos_weight']:.2f} "
          f"| effective_swing_threshold={stats['effective_swing_threshold']:.2%}")

    if category in CALIBRATED_DECISION_THRESHOLDS:
        decision_threshold = CALIBRATED_DECISION_THRESHOLDS[category]
        model_registry.update_decision_threshold(category, decision_threshold)
        print(f"Applied calibrated decision_threshold={decision_threshold:.2%} "
              f"(computed by scripts/expected_value_thresholds.py)")

    # Fresh detector loaded straight from the signed manifest, exactly like a real user's
    # session would -- this also exercises the integrity check end to end.
    provider = AlphaVantageProvider(api_key=None)
    detector = SwingTradeDetector(category, provider)
    if not detector.is_ready:
        raise NoModelAvailableError(f"Just-trained category '{category}' failed to reload")

    metrics = classification_metrics_on_test_split(category, detector)
    if metrics is None:
        print("Out-of-sample test-split classification: no readable test files")
    else:
        beats = "BEATS" if metrics["accuracy"] > metrics["majority_accuracy"] else "does NOT beat"
        lift = metrics["pr_auc"] / metrics["base_rate"] if metrics["base_rate"] else float("nan")
        print(f"Out-of-sample test-split classification ({metrics['rows']} rows, "
              f"base rate {metrics['base_rate']:.2%}):")
        print(f"  PR-AUC={metrics['pr_auc']:.4f} ({lift:.1f}x base rate)  "
              f"ROC-AUC={metrics['roc_auc']:.4f}  "
              f"precision={metrics['precision']:.1%}  recall={metrics['recall']:.1%}  "
              f"signals={metrics['signals']}")
        print(f"  accuracy={metrics['accuracy']:.1%} vs. {metrics['majority_accuracy']:.1%} for "
              f"always-negative -- {beats} the baseline")

    combined, per_symbol = evaluate_on_test_split(category, detector)
    null_combined, _ = evaluate_on_test_split(category, detector, decision_threshold=0.0)
    null = compare_against_null(combined, null_combined)
    print(f"Model-off null (same entries/exits, ranking ignored): {null_combined['num_trades']} trades, "
          f"win_rate={null_combined['win_rate']:.1%}, avg_profit={null_combined['avg_profit']:.2%}")
    print(f"  edge over null: {null['edge']:+.2%}/trade "
          f"({null['edge_in_standard_errors']:+.2f} standard errors) -- {null['verdict']}")
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
        "validation_pr_auc": stats["validation_pr_auc"],
        "scale_pos_weight": stats["scale_pos_weight"],
        "test_base_rate": metrics["base_rate"] if metrics else float("nan"),
        "test_pr_auc": metrics["pr_auc"] if metrics else float("nan"),
        "test_roc_auc": metrics["roc_auc"] if metrics else float("nan"),
        "test_precision": metrics["precision"] if metrics else float("nan"),
        "test_recall": metrics["recall"] if metrics else float("nan"),
        "test_beats_baseline": (
            bool(metrics["accuracy"] > metrics["majority_accuracy"]) if metrics else False
        ),
        "test_null_trades": null_combined["num_trades"],
        "test_null_avg_profit": null_combined["avg_profit"],
        "test_edge_over_null": null["edge"],
        "test_edge_std_errors": null["edge_in_standard_errors"],
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
