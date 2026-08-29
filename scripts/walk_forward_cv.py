"""Purged walk-forward cross-validation: error bars for every per-category number.

Every figure this project reports today comes from one train/validation split and one
test window. That makes two different questions indistinguishable -- "this model has no
usable decision threshold" and "one three-year window wasn't enough to find one" -- and
it is why scripts/select_thresholds.py currently declines to pick for five of the eight
categories. A 29-trade sample is not evidence either way.

This runs the split five times instead. Folds are expanding-window and anchored: fold k
trains on everything up to cutoff k and validates on the block that follows it, so each
fold is a genuine forecast of a later period and later folds train on more history --
the same shape as deployment, unlike k-fold, which would train on the future.

    fold 1  |=== train ===|# val #|
    fold 2  |===== train ======|# val #|
    fold 3  |======== train =======|# val #|             # = purged embargo
    ...

Three properties this preserves from the rest of the pipeline:

  * test/ is never touched. Folds are cut from train/ + validation/ only, so the final
    holdout stays unused and the numbers here can inform threshold choice without
    spending it.
  * Cutoffs are category-wide calendar dates, not per-symbol row counts, for the reason
    in scripts/build_factor_datasets.py -- a category's symbols are near-duplicates, so
    a per-symbol cut leaks one symbol's future into another's past.
  * Each fold boundary carries the same embargo, because a swing label reads ten bars
    forward.

What it reports, per category: PR-AUC and ROC-AUC as mean +/- std across folds rather
than a single number; the same edge-over-null comparison scripts/train_all_categories.py
prints, per fold, so a category that beats the null in one regime and loses in four is
visible as such; and a threshold sweep pooled across all five folds, which is the point
-- pooling multiplies the trade count at every candidate and lets the selection rule in
scripts/select_thresholds.py work with a sample that can actually support it.

Selection across folds is stricter than selection on one window, in two ways that the
single-split rule in scripts/select_thresholds.py has no way to express:

  * The trade floor scales with the fold count. Pooling five folds and then applying a
    20-trade floor would be a *weaker* bar than the single-split rule, not a stronger
    one -- 20 pooled trades is four per fold. The floor is MIN_TRADES_PER_FOLD times the
    number of folds that actually produced trades.
  * A candidate must work in most folds, not merely in aggregate. Pooling hides the
    threshold that is spectacular in one regime and negative in four, which is precisely
    the failure mode cross-validation exists to expose, so a candidate also has to show
    a positive mean in at least FOLD_AGREEMENT of the folds where it traded.

Raw per-fold, per-threshold trade returns are written to the JSON named by
WALK_FORWARD_CV_OUTPUT, so the selection rule can be revised and re-derived without
paying for the 40 model fits again.

Cost: one model fit per category per fold (40 by default). Budget roughly 45 minutes.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (  # noqa: E402
    CALIBRATED_SWING_THRESHOLDS,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_MIN_HOLD_PERIODS,
    DEFAULT_SWING_THRESHOLD,
)
from app.data_loader import (  # noqa: E402
    DataProcessor,
    category_train_dir,
    category_validation_dir,
    list_categories,
)
from app.detector import SwingTradeDetector, score_for_backtest, simulate_trades  # noqa: E402
from app.indicators import TechnicalIndicators  # noqa: E402
from app.labeling import create_swing_labels, effective_threshold  # noqa: E402
from app.market_context import load_context  # noqa: E402
from app.trainer import SwingTradeTrainer  # noqa: E402
from scripts.build_factor_datasets import split_by_cutoffs  # noqa: E402
from scripts.select_thresholds import SHRINKAGE, THRESHOLDS  # noqa: E402

DEFAULT_FOLDS = 5
MIN_FOLD_TRAIN_ROWS = 500  # SwingTradeTrainer's own floor
MIN_FOLD_VALIDATION_ROWS = 250

# Per-fold trade floor. The pooled requirement is this times the number of folds that
# traded, so adding folds raises the bar instead of lowering it.
MIN_TRADES_PER_FOLD = 20
# Share of trading folds in which a candidate must show a positive mean return.
FOLD_AGREEMENT = 0.6


class FoldDetector:
    """Duck-types the part of SwingTradeDetector that the backtest uses, around a model
    that was just fitted in memory.

    CV fits 40 models and none of them should be written to models/ -- they are
    measurements, not artifacts, and persisting them would overwrite the real ones.
    calculate_stop_take_profit is borrowed from the real class rather than reimplemented
    so a fold's exits match production exactly.
    """

    calculate_stop_take_profit = SwingTradeDetector.calculate_stop_take_profit

    def __init__(self, category, trainer, decision_threshold):
        self.category = category
        self.is_ready = True
        # Declared for app.detector.resolve_market_context: a fold's model was trained
        # with whatever context its trainer had, and scoring it must use the same.
        self.market_context = trainer.market_context
        self.uses_market_context = trainer.market_context is not None
        self.model = trainer.model
        self.scaler = trainer.scaler
        self.feature_columns = trainer.feature_columns
        self.decision_threshold = decision_threshold
        self.lookforward_periods = trainer.lookforward_periods
        self.effective_swing_threshold = effective_threshold(trainer.swing_threshold)


def load_pre_test_history(category):
    """Everything before the test cutoff, per symbol: train/ plus validation/.

    test/ is deliberately not read. Folds cut only from what the pipeline has already
    designated as pre-test, so cross-validating here spends none of the final holdout.
    """
    histories = {}
    for directory in (category_train_dir(category), category_validation_dir(category)):
        if not os.path.isdir(directory):
            continue
        for filename in sorted(f for f in os.listdir(directory) if f.endswith(".csv")):
            symbol = DataProcessor.infer_symbol_from_path(filename).upper()
            try:
                frame = DataProcessor.load_and_validate_data(os.path.join(directory, filename))
            except Exception:
                continue
            frame = frame.drop(columns=["symbol"], errors="ignore")
            histories[symbol] = (
                pd.concat([histories[symbol], frame]).sort_index() if symbol in histories else frame
            )
    return {symbol: frame[~frame.index.duplicated(keep="first")] for symbol, frame in histories.items()}


def make_fold_cutoffs(histories, n_folds=DEFAULT_FOLDS):
    """(train_end, validation_end) per fold, expanding-window, on the category calendar.

    Boundaries are quantiles of the pooled trading days from the median symbol's first
    date onward -- the same restriction scripts/build_factor_datasets.py uses, and for
    the same reason: one long history would otherwise place every boundary in an era
    when most of the category's symbols did not yet exist.
    """
    first_dates = sorted(frame.index.min() for frame in histories.values())
    common_start = first_dates[len(first_dates) // 2]
    pooled = pd.DatetimeIndex(
        [date for frame in histories.values() for date in frame.index if date >= common_start]
    ).sort_values()
    if pooled.empty:
        raise ValueError("no rows at or after the category's common start date")

    # n_folds + 1 blocks: block 0 is train-only seed history, then each fold validates on
    # the next block and trains on everything before it.
    edges = [pooled[min(int(len(pooled) * i / (n_folds + 1)), len(pooled) - 1)] for i in range(n_folds + 2)]
    return [(edges[i + 1], edges[i + 2]) for i in range(n_folds)]


def write_fold_frames(histories, train_end, validation_end, directory):
    """Splits each symbol at the fold's cutoffs (embargoed) and writes the train side.

    SwingTradeTrainer.train takes a directory, so the fold's training frames go to a
    temporary one. Returns the validation frames in memory -- they are only ever scored,
    never re-read.
    """
    os.makedirs(directory, exist_ok=True)
    validation_frames = {}
    train_rows = 0
    for symbol, frame in histories.items():
        parts = split_by_cutoffs(frame, train_end, validation_end)
        if not parts["train"].empty:
            parts["train"].to_csv(os.path.join(directory, f"{symbol}.csv"))
            train_rows += len(parts["train"])
        if not parts["validation"].empty:
            validation_frames[symbol] = parts["validation"]
    return validation_frames, train_rows


def classification_metrics(detector, validation_frames, swing_threshold, lookforward):
    """PR-AUC and friends on the fold's validation block, labelled the way the model was
    trained to predict."""
    context = load_context()
    frames = []
    for frame in validation_frames.values():
        features = TechnicalIndicators.create_all_indicators(frame, market_context=context)
        labeled_frame = create_swing_labels(features, swing_threshold, lookforward, DEFAULT_MIN_HOLD_PERIODS)
        # Filled per symbol and forward only, before the concat -- the same discipline as
        # app/trainer.py. A frame-wide fill lets one symbol supply another's missing
        # columns, and a back-fill writes later values into earlier rows.
        fill_cols = [c for c in labeled_frame.columns if c != "swing_label"]
        labeled_frame[fill_cols] = labeled_frame[fill_cols].ffill()
        frames.append(labeled_frame)
    if not frames:
        return None

    labeled = pd.concat(frames)
    for feature in detector.feature_columns:
        if feature not in labeled.columns:
            labeled[feature] = np.nan
    X = labeled[detector.feature_columns].replace([np.inf, -np.inf], 0).fillna(0)
    y = labeled["swing_label"].astype(int).to_numpy()
    if len(y) == 0:
        return None

    probabilities = detector.model.predict_proba(detector.scaler.transform(X))[:, 1]
    predictions = (probabilities >= detector.decision_threshold).astype(int)
    base_rate = float(y.mean())
    both_classes = 0 < base_rate < 1
    return {
        "rows": len(y),
        "base_rate": base_rate,
        "pr_auc": float(average_precision_score(y, probabilities)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else float("nan"),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
    }


def sweep_and_null(detector, validation_frames):
    """Scores the fold's validation block once, then simulates every candidate threshold
    plus the model-off null over it. Returns {threshold: [profits]} and the null profits."""
    scored = {}
    for symbol, frame in validation_frames.items():
        try:
            scored[symbol] = score_for_backtest(detector, frame)
        except ValueError:
            continue  # too few rows for a full hold in this fold's block

    per_threshold = {threshold: [] for threshold in THRESHOLDS}
    null_profits = []
    for scoring in scored.values():
        for threshold in THRESHOLDS:
            result = simulate_trades(detector, scoring, decision_threshold=threshold)
            per_threshold[threshold].extend(trade["profit_pct"] for trade in result["trades"])
        null_result = simulate_trades(detector, scoring, decision_threshold=0.0)
        null_profits.extend(trade["profit_pct"] for trade in null_result["trades"])
    return per_threshold, null_profits


def summarize(profits):
    """The same shrunk score scripts/select_thresholds.py selects on, over a pooled
    sample: mean per-trade return minus one standard error."""
    array = np.asarray(profits, dtype=float)
    count = len(array)
    if count == 0:
        return {"num_trades": 0, "mean_profit": 0.0, "standard_error": 0.0,
                "score": float("-inf"), "win_rate": 0.0, "sharpe": 0.0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    standard_error = std / np.sqrt(count) if count > 1 else float("inf")
    return {
        "num_trades": count,
        "mean_profit": mean,
        "standard_error": standard_error if count > 1 else 0.0,
        "score": (mean - SHRINKAGE * standard_error) if count > 1 else float("-inf"),
        "win_rate": float((array > 0).mean()),
        "sharpe": float(mean / array.std() * (12 ** 0.5)) if array.std() else 0.0,
    }


def select_across_folds(pooled_sweep, per_fold_profits, default_threshold):
    """The single-window rule, plus the two fold-aware requirements in the docstring.

    per_fold_profits maps threshold -> [profits per fold]. Returns (threshold, note).
    """
    trading_folds = {
        threshold: [profits for profits in folds if profits]
        for threshold, folds in per_fold_profits.items()
    }

    annotated = []
    for row in pooled_sweep:
        folds = trading_folds.get(row["threshold"], [])
        agreeing = sum(1 for profits in folds if float(np.mean(profits)) > 0)
        needed = max(1, int(np.ceil(FOLD_AGREEMENT * len(folds)))) if folds else 1
        floor = MIN_TRADES_PER_FOLD * max(1, len(folds))
        annotated.append({
            **row,
            "trading_folds": len(folds),
            "agreeing_folds": agreeing,
            "folds_needed": needed,
            "trade_floor": floor,
            "eligible": (row["num_trades"] >= floor and row["score"] > 0 and agreeing >= needed),
        })

    eligible = [row for row in annotated if row["eligible"]]
    if not eligible:
        blocked = [r for r in annotated if r["num_trades"] >= r["trade_floor"] and r["score"] > 0]
        detail = (f"{len(blocked)} cleared the scaled trade floor with a positive score but "
                  f"failed fold agreement" if blocked else
                  "none cleared the scaled trade floor with a positive score")
        return default_threshold, annotated, f"no candidate survived cross-validation ({detail}) -- kept the deployed value"

    best = max(eligible, key=lambda row: row["score"])
    position = annotated.index(best)
    lower = annotated[position - 1] if position > 0 else None
    upper = annotated[position + 1] if position + 1 < len(annotated) else None
    if lower is None or upper is None:
        return default_threshold, annotated, (
            f"best score is at {best['threshold']:.0%}, the edge of the swept grid -- kept the deployed value")
    if upper["score"] > best["score"] or lower["score"] > best["score"]:
        return default_threshold, annotated, (
            f"best eligible score at {best['threshold']:.0%} is not an interior peak "
            f"(a neighbour scores higher and was excluded by a constraint) -- kept the deployed value")

    return best["threshold"], annotated, (
        f"interior peak: {best['num_trades']} pooled trades (floor {best['trade_floor']}), "
        f"positive in {best['agreeing_folds']}/{best['trading_folds']} trading folds "
        f"(needed {best['folds_needed']}), score {best['score']:+.2%}")


def run_category(category, n_folds=DEFAULT_FOLDS):
    histories = load_pre_test_history(category)
    if not histories:
        raise ValueError("no pre-test history on disk")
    swing_threshold = CALIBRATED_SWING_THRESHOLDS.get(category, DEFAULT_SWING_THRESHOLD)
    # The threshold currently in production, used as the per-fold operating point and as
    # the fallback when the pooled sweep finds nothing. Falls back to the global default
    # when the category has never been trained -- CV does not require a saved model.
    try:
        deployed_threshold = SwingTradeDetector(category, provider=None).decision_threshold
    except Exception:
        deployed_threshold = DEFAULT_DECISION_THRESHOLD

    folds = []
    pooled_by_threshold = {threshold: [] for threshold in THRESHOLDS}
    per_fold_by_threshold = {threshold: [] for threshold in THRESHOLDS}
    pooled_null = []

    for index, (train_end, validation_end) in enumerate(make_fold_cutoffs(histories, n_folds), start=1):
        workspace = tempfile.mkdtemp(prefix=f"wfcv_{category}_")
        try:
            validation_frames, train_rows = write_fold_frames(
                histories, train_end, validation_end, workspace
            )
            validation_rows = sum(len(frame) for frame in validation_frames.values())
            label = f"  fold {index}/{n_folds}  train < {train_end.date()} | val < {validation_end.date()}"
            if train_rows < MIN_FOLD_TRAIN_ROWS or validation_rows < MIN_FOLD_VALIDATION_ROWS:
                print(f"{label}  SKIPPED (train {train_rows} rows, validation {validation_rows})")
                continue

            trainer = SwingTradeTrainer(swing_threshold=swing_threshold)
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):  # the trainer's own report, not ours
                trainer.train(workspace)
            detector = FoldDetector(category, trainer, deployed_threshold)

            metrics = classification_metrics(
                detector, validation_frames, swing_threshold, trainer.lookforward_periods
            )
            per_threshold, null_profits = sweep_and_null(detector, validation_frames)
            for threshold, profits in per_threshold.items():
                pooled_by_threshold[threshold].extend(profits)
                per_fold_by_threshold[threshold].append(list(profits))
            pooled_null.extend(null_profits)

            deployed = summarize(per_threshold.get(_nearest(deployed_threshold), []))
            null = summarize(null_profits)
            edge = deployed["mean_profit"] - null["mean_profit"]
            # NaN, not 0.0, when the fold produced too few trades at the deployed
            # threshold to have a standard error: "couldn't measure" and "measured no
            # edge" are different answers and must not print the same way.
            edge_se = (edge / deployed["standard_error"]) if deployed["standard_error"] else float("nan")
            folds.append({
                "fold": index, "train_end": train_end, "validation_end": validation_end,
                "train_rows": train_rows, "validation_rows": validation_rows,
                **(metrics or {}), "deployed": deployed, "null": null, "edge_std_errors": edge_se,
            })
            edge_text = f"{edge:+.2%} ({edge_se:+.2f} SE)" if edge_se == edge_se else (
                f"n/a ({deployed['num_trades']} trades at {deployed_threshold:.0%})")
            print(f"{label}  PR-AUC {metrics['pr_auc']:.4f}  ROC-AUC {metrics['roc_auc']:.4f}  "
                  f"base {metrics['base_rate']:.2%}  |  {deployed['num_trades']:3d} trades  |  "
                  f"edge over null {edge_text}")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    if not folds:
        raise ValueError("no fold had enough data to train and evaluate")

    pooled_sweep = [{"threshold": t, **summarize(p)} for t, p in sorted(pooled_by_threshold.items())]
    selected, annotated_sweep, note = select_across_folds(
        pooled_sweep, per_fold_by_threshold, deployed_threshold
    )
    return {
        "category": category,
        "swing_threshold": swing_threshold,
        "deployed_threshold": deployed_threshold,
        "folds": folds,
        "pooled_sweep": annotated_sweep,
        "per_fold_profits": {str(t): p for t, p in per_fold_by_threshold.items()},
        "pooled_null": summarize(pooled_null),
        "selected_threshold": selected,
        "selection_note": note,
    }


def _nearest(threshold):
    return min(THRESHOLDS, key=lambda candidate: abs(candidate - threshold))


def _mean_std(values):
    clean = [v for v in values if v == v]  # drop NaN folds
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), (float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0)


def main(n_folds=DEFAULT_FOLDS):
    summaries = []
    failures = []
    raw_results = []
    for category in list_categories():
        print(f"\n{'=' * 82}\n{category}\n{'=' * 82}")
        try:
            result = run_category(category, n_folds)
        except Exception as error:
            print(f"  FAILED: {error}")
            failures.append((category, str(error)))
            continue

        raw_results.append(result)
        pr_mean, pr_std = _mean_std([f.get("pr_auc", float("nan")) for f in result["folds"]])
        roc_mean, roc_std = _mean_std([f.get("roc_auc", float("nan")) for f in result["folds"]])
        edges = [f["edge_std_errors"] for f in result["folds"]]
        measurable = [e for e in edges if e == e]  # folds with enough trades to compare
        beat_null = sum(1 for e in measurable if e > 0)
        pooled_at_selected = next(
            row for row in result["pooled_sweep"] if row["threshold"] == _nearest(result["selected_threshold"])
        )

        print(f"\n  across {len(result['folds'])} folds: "
              f"PR-AUC {pr_mean:.4f} +/- {pr_std:.4f}   ROC-AUC {roc_mean:.4f} +/- {roc_std:.4f}")
        print(f"  beat the model-off null in {beat_null}/{len(measurable)} measurable folds "
              f"({len(edges) - len(measurable)} had too few trades to compare) "
              f"-- edge in SE per fold: "
              f"{', '.join(f'{e:+.2f}' if e == e else 'n/a' for e in edges)}")
        print(f"  pooled threshold sweep ({sum(r['num_trades'] for r in result['pooled_sweep'][:1])} trades "
              f"at the loosest candidate): selected {result['selected_threshold']:.0%} "
              f"(deployed {result['deployed_threshold']:.0%})")
        print(f"    {result['selection_note']}")

        summaries.append({
            "category": category,
            "folds": len(result["folds"]),
            "pr_auc_mean": pr_mean, "pr_auc_std": pr_std,
            "roc_auc_mean": roc_mean, "roc_auc_std": roc_std,
            "folds_beating_null": beat_null,
            "folds_measurable": len(measurable),
            "deployed": result["deployed_threshold"],
            "cv_selected": result["selected_threshold"],
            "pooled_trades_at_selected": pooled_at_selected["num_trades"],
            "pooled_mean_at_selected": pooled_at_selected["mean_profit"],
            "pooled_score_at_selected": pooled_at_selected["score"],
        })

    print(f"\n{'=' * 82}\nSummary\n{'=' * 82}")
    if summaries:
        print(pd.DataFrame(summaries).to_string(index=False))

    out_path = os.environ.get("WALK_FORWARD_CV_OUTPUT")
    if out_path:
        import json
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(raw_results, handle, indent=2, default=str)
        print(f"\nPer-fold detail written to {out_path}")
    if failures:
        print("\nFailures:")
        for category, error in failures:
            print(f"  {category}: {error}")
    return summaries, failures


if __name__ == "__main__":
    folds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FOLDS
    _, cv_failures = main(folds)
    sys.exit(1 if cv_failures else 0)
