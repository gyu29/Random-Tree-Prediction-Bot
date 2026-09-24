"""Regenerates every table and figure for the write-up, from one frozen dataset.

Everything the paper reports comes out of this script and nothing else. It exists because
the numbers had drifted into comments in app/config.py, several of them measured at a
horizon this project no longer uses, and a result that cannot be regenerated on demand
cannot be checked by anyone -- including the author a month later.

What makes the run reproducible:

  * The data is frozen. `--freeze-data` records a SHA-256 for every CSV under train/,
    validation/ and test/ and for market_context.csv, in paper/data_manifest.json. Every
    later run verifies against it and refuses to start on a mismatch. Yahoo revises
    adjusted prices after every dividend, so re-downloading is not the same dataset, and
    this is the only way to know two runs saw the same numbers.
  * The models are refitted here, in memory, the same way app/trading_system.py fits them
    -- same trainer, same hyperparameters, same seeds -- and never written to models/.
    The saved models are only read, to report whether the refit reproduces them.
  * The evaluation rules are the project's own, called rather than copied:
    scripts/expected_value_thresholds.solve_threshold picks each floor on validation/,
    and test/ is scored once, at that floor, against the model-off null. Nothing here
    chooses a threshold, a block length or a bar.
  * The run records the commit, whether the working tree was clean, every library
    version and every seed in run_manifest.json. `--require-clean` refuses a dirty tree;
    use it for the run whose numbers go in the paper.

Descriptive statistics added on top of the project's gate, and labelled as such: one-sided
p-values for the out-of-sample edge, and the Bonferroni bar for having tested eight
categories. They report how far a result is from each bar. They decide nothing.

Usage:
    python scripts/paper_run.py --freeze-data        # once, to pin the dataset
    python scripts/paper_run.py --require-clean      # the run the paper cites
    python scripts/paper_run.py --folds 0 --categories small_cap --out /tmp/trial

Cost: one model fit per category, plus one per category per cross-validation fold.
About 17 minutes with the default five folds on one machine; --folds 0 skips cross-validation.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from app.config import (  # noqa: E402
    CALIBRATED_DECISION_THRESHOLDS,
    CALIBRATED_SWING_THRESHOLDS,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_LOOKFORWARD_PERIODS,
    DEFAULT_MIN_HOLD_PERIODS,
    DEFAULT_SWING_THRESHOLD,
    FACTOR_CATEGORIES,
    LABEL_MODE,
    MARKET_CONTEXT_PATH,
    TEST_ROOT,
    TRAIN_ROOT,
    VALIDATION_ROOT,
)
from app.data_loader import (  # noqa: E402
    category_test_dir,
    category_train_dir,
    category_validation_dir,
    list_categories,
)
from app.detector import SwingTradeDetector, resolve_market_context, simulate_trades  # noqa: E402
from app.indicators import TechnicalIndicators  # noqa: E402
from app.labeling import create_swing_labels  # noqa: E402
from app.market_data.alpha_vantage_provider import AlphaVantageProvider  # noqa: E402
from app.trainer import SwingTradeTrainer  # noqa: E402
from scripts import walk_forward_cv  # noqa: E402
from scripts.build_factor_datasets import EMBARGO_PERIODS, _required_sessions  # noqa: E402
from scripts.expected_value_thresholds import (  # noqa: E402
    BLOCK_LENGTH_CALENDAR_DAYS,
    BOOTSTRAP_REPLICATES,
    MIN_BLOCKS_FOR_RESAMPLING,
    PERMUTATION_SEED,
    PERMUTATIONS,
    date_blocks,
    null_trades,
    paired_date_blocks,
    solve_threshold,
)
from scripts.select_thresholds import load_symbol_data, score_once  # noqa: E402
from scripts.train_all_categories import compare_against_null  # noqa: E402

PAPER_ROOT = os.path.join(ROOT, "paper")
DATA_MANIFEST_PATH = os.path.join(PAPER_ROOT, "data_manifest.json")
DEFAULT_OUT = os.path.join(PAPER_ROOT, "results")

# The bar compare_against_null in scripts/train_all_categories.py calls "real edge".
PROJECT_BAR = 2.0
# Descriptive only: the same two-sided 5% as PROJECT_BAR, divided across every category
# the project tested. A single category clearing PROJECT_BAR is one of eight looks.
FAMILY_ALPHA = 0.05

# The three holds this project has used or measured, as (label, min_hold, lookforward).
HORIZONS = [
    ("3-10 sessions (original)", 3, 10),
    ("63-126 sessions (current)", DEFAULT_MIN_HOLD_PERIODS, DEFAULT_LOOKFORWARD_PERIODS),
    ("126-252 sessions (rejected)", 126, 252),
]
# The per-symbol split build_factor_datasets.py replaced, measured for the leakage table.
PER_SYMBOL_FRACTIONS = (0.55, 0.15)
RELIABILITY_BINS = 10

# Print-friendly: white surface, recessive axes. Categorical order from the dataviz
# reference palette -- slot 1 is the primary series, slot 2 the comparison.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, RULE = "#0b0b0b", "#52514e", "#bdbcb6"


# ----------------------------------------------------------------------------- data


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_files():
    files = []
    for root in (TRAIN_ROOT, VALIDATION_ROOT, TEST_ROOT):
        for category in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            directory = os.path.join(root, category)
            files += [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith(".csv")]
    files.append(MARKET_CONTEXT_PATH)
    return files


def _split_cutoffs():
    """Each category's first validation and first test date, read off the files."""
    cutoffs = {}
    for category in list_categories():
        firsts = {}
        for name, directory in (("validation_start", category_validation_dir(category)),
                                ("test_start", category_test_dir(category))):
            frames = load_symbol_data(directory)
            firsts[name] = str(min(frame.index.min() for frame in frames.values()).date()) if frames else None
        cutoffs[category] = firsts
    return cutoffs


def freeze_data():
    files = _data_files()
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": ("The dataset every paper number is computed from. scripts/paper_run.py "
                 "refuses to run if any file differs."),
        "embargo_rows_per_seam": EMBARGO_PERIODS,
        "cutoffs": _split_cutoffs(),
        "files": {os.path.relpath(path, ROOT): _sha256(path) for path in files},
    }
    os.makedirs(PAPER_ROOT, exist_ok=True)
    with open(DATA_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    print(f"Froze {len(files)} files into {os.path.relpath(DATA_MANIFEST_PATH, ROOT)}")


class DataMismatch(Exception):
    pass


def verify_data():
    if not os.path.exists(DATA_MANIFEST_PATH):
        raise DataMismatch("no paper/data_manifest.json -- run with --freeze-data first")
    with open(DATA_MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = manifest["files"]
    actual = {os.path.relpath(path, ROOT): path for path in _data_files()}
    problems = [f"missing: {p}" for p in sorted(set(expected) - set(actual))]
    problems += [f"not in manifest: {p}" for p in sorted(set(actual) - set(expected))]
    problems += [f"changed: {p}" for p in sorted(set(expected) & set(actual))
                 if _sha256(actual[p]) != expected[p]]
    if problems:
        shown = "\n  ".join(problems[:15])
        more = f"\n  ... and {len(problems) - 15} more" if len(problems) > 15 else ""
        raise DataMismatch(f"data on disk is not the frozen dataset:\n  {shown}{more}")
    return manifest


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance():
    import joblib
    import sklearn
    import xgboost
    from importlib.metadata import version

    # Dirty means something that could change a number: any modified tracked file, or an
    # untracked .py file that could be imported. Untracked notes and scratch data cannot.
    changed = [line for line in (_git("status", "--porcelain", "--untracked-files=all") or "").splitlines()
               if not line.startswith("??") or line.rstrip("/").endswith(".py")]
    return {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(changed),
        "git_changed_files": changed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__, "ta": version("ta"),
            "matplotlib": matplotlib.__version__,
        },
        "seeds": {"random_forest": 42, "xgboost": 42, "permutation_and_bootstrap": PERMUTATION_SEED},
        "settings": {
            "label_mode": LABEL_MODE,
            "min_hold_periods": DEFAULT_MIN_HOLD_PERIODS,
            "lookforward_periods": DEFAULT_LOOKFORWARD_PERIODS,
            "block_length_calendar_days": BLOCK_LENGTH_CALENDAR_DAYS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutations": PERMUTATIONS,
            "min_blocks_for_resampling": MIN_BLOCKS_FOR_RESAMPLING,
            "project_bar_standard_errors": PROJECT_BAR,
            "family_alpha_two_sided": FAMILY_ALPHA,
            "swing_thresholds": {c: CALIBRATED_SWING_THRESHOLDS.get(c, DEFAULT_SWING_THRESHOLD)
                                 for c in list_categories()},
        },
    }


def _strip_symbol(frames):
    return {s: f.drop(columns=["symbol"], errors="ignore") for s, f in frames.items()}


def split_frames(category):
    return {
        "train": _strip_symbol(load_symbol_data(category_train_dir(category))),
        "validation": _strip_symbol(load_symbol_data(category_validation_dir(category))),
        "test": _strip_symbol(load_symbol_data(category_test_dir(category))),
    }


def full_histories(splits):
    """Each symbol's three files joined back together. The embargo rows at each seam were
    never written to disk, so these histories have two 126-session gaps."""
    joined = {}
    for frames in splits.values():
        for symbol, frame in frames.items():
            joined[symbol] = pd.concat([joined[symbol], frame]) if symbol in joined else frame
    return {s: f.sort_index()[~f.sort_index().index.duplicated()] for s, f in joined.items()}


# ------------------------------------------------------------ data-only measurements


def effective_series(splits):
    """n / (1 + (n - 1) * mean pairwise correlation) of daily returns.

    Returns are taken within each split file and then joined, so no return spans the
    embargo gap -- one six-month move per symbol would otherwise sit in every covariance.
    """
    returns = {}
    for frames in splits.values():
        for symbol, frame in frames.items():
            series = frame["close"].pct_change().dropna()
            returns[symbol] = pd.concat([returns[symbol], series]) if symbol in returns else series
    table = pd.DataFrame({s: r[~r.index.duplicated()] for s, r in returns.items()})
    correlation = table.corr(min_periods=250).to_numpy()
    n = correlation.shape[0]
    off_diagonal = correlation[~np.eye(n, dtype=bool)]
    mean_correlation = float(np.nanmean(off_diagonal)) if n > 1 else 1.0
    return n / (1 + (n - 1) * mean_correlation), mean_correlation


def leakage(splits):
    """Share of test rows dated on a day some *other* symbol in the category trained on.

    Measured twice: under the per-symbol percentage split this project used to use, and
    under the category-wide calendar split on disk now.
    """
    def share(assignments):
        train_dates = {s: set(parts["train"].index) for s, parts in assignments.items()}
        leaked = total = 0
        for symbol, parts in assignments.items():
            siblings = set().union(*(d for s, d in train_dates.items() if s != symbol)) if len(train_dates) > 1 else set()
            dates = parts["test"].index
            total += len(dates)
            leaked += sum(1 for date in dates if date in siblings)
        return leaked / total if total else float("nan")

    per_symbol = {}
    for symbol, frame in full_histories(splits).items():
        n = len(frame)
        train_end = int(n * PER_SYMBOL_FRACTIONS[0])
        validation_end = int(n * sum(PER_SYMBOL_FRACTIONS))
        per_symbol[symbol] = {"train": frame.iloc[:train_end], "test": frame.iloc[validation_end:]}
    empty = pd.DataFrame(index=pd.DatetimeIndex([]))
    calendar = {s: {"train": splits["train"].get(s, empty), "test": splits["test"].get(s, empty)}
                for s in set(splits["train"]) | set(splits["test"])}
    return share(per_symbol), share(calendar)


def label_rates(train_frames, swing_threshold, min_hold, lookforward):
    """Positive-label rate on train/, terminal and peak, over rows with a full window."""
    rates = {}
    for mode in ("terminal", "peak"):
        positives = rows = 0
        for frame in train_frames.values():
            if len(frame) <= lookforward:
                continue
            labels = create_swing_labels(frame, swing_threshold, lookforward, min_hold, mode=mode)
            usable = labels["swing_label"].iloc[:-lookforward]
            positives += int(usable.sum())
            rows += len(usable)
        rates[mode] = positives / rows if rows else float("nan")
    return rates


def coverage_sessions(splits):
    """Sessions from the category's common start (median symbol's first date) to its end,
    counting back the two embargoed seams that are not on disk."""
    histories = full_histories(splits)
    firsts = sorted(frame.index.min() for frame in histories.values())
    common_start = firsts[len(firsts) // 2]
    dates = {d for frame in histories.values() for d in frame.index if d >= common_start}
    return len(dates) + 2 * EMBARGO_PERIODS, common_start


# ------------------------------------------------------------------ model evaluation


def labeled_rows(detector, frames, swing_threshold):
    """Every scorable row with a known outcome: features complete, full label window."""
    context = resolve_market_context(detector)
    X_parts, y_parts = [], []
    for frame in frames.values():
        features = TechnicalIndicators.create_all_indicators(frame, market_context=context)
        labeled = create_swing_labels(features, swing_threshold, detector.lookforward_periods,
                                      detector.min_hold_periods, mode=LABEL_MODE)
        fill_cols = [c for c in labeled.columns if c != "swing_label"]
        labeled[fill_cols] = labeled[fill_cols].ffill()
        for feature in detector.feature_columns:
            if feature not in labeled.columns:
                labeled[feature] = 0.0
        labeled = labeled.iloc[:-detector.lookforward_periods]
        X = labeled[detector.feature_columns].replace([np.inf, -np.inf], 0)
        keep = X.notna().all(axis=1)
        X_parts.append(X[keep])
        y_parts.append(labeled.loc[keep, "swing_label"].astype(int))
    if not X_parts:
        return None, None
    X, y = pd.concat(X_parts), pd.concat(y_parts).to_numpy()
    return detector.model.predict_proba(detector.scaler.transform(X))[:, 1], y


def classification(probabilities, labels):
    base_rate = float(labels.mean())
    both = 0 < base_rate < 1
    pr_auc = float(average_precision_score(labels, probabilities)) if both else float("nan")
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, RELIABILITY_BINS + 1)))
    reliability = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        inside = (probabilities >= lower) & ((probabilities <= upper) if upper == edges[-1] else (probabilities < upper))
        if inside.any():
            reliability.append({"predicted": float(probabilities[inside].mean()),
                                "observed": float(labels[inside].mean()), "rows": int(inside.sum())})
    return {
        "rows": len(labels), "base_rate": base_rate, "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / base_rate if base_rate else float("nan"),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both else float("nan"),
        "ece": SwingTradeTrainer._expected_calibration_error(labels, probabilities),
        "reliability": reliability,
    }


def trades_at(detector, scored, threshold):
    trades = []
    for symbol, scoring in scored.items():
        for trade in simulate_trades(detector, scoring, decision_threshold=threshold)["trades"]:
            trades.append({**trade, "symbol": symbol})
    return trades


def combined(trades):
    """The dict shape compare_against_null in scripts/train_all_categories.py reads."""
    profits = np.asarray([t["profit_pct"] for t in trades], dtype=float)
    return {
        "num_trades": len(trades),
        "profits": profits,
        "entry_dates": np.asarray([pd.Timestamp(t["entry_date"]).normalize() for t in trades]),
        "win_rate": float((profits > 0).mean()) if len(profits) else 0.0,
        "avg_profit": float(profits.mean()) if len(profits) else 0.0,
        "std_profit": float(profits.std(ddof=1)) if len(profits) > 1 else 0.0,
    }


def test_against_null(detector, test_scored, threshold, null_trade_list, trades_path):
    """The model's test trades at `threshold` against the model-off null, through the same
    compare_against_null scripts/train_all_categories.py reports."""
    model_trades = trades_at(detector, test_scored, threshold)
    pd.DataFrame(model_trades).to_csv(trades_path, index=False)
    model, null = combined(model_trades), combined(null_trade_list)
    comparison = compare_against_null(model, null)
    naive = (float(np.sqrt(model["profits"].var(ddof=1) / model["num_trades"]
                           + null["profits"].var(ddof=1) / null["num_trades"]))
             if model["num_trades"] > 1 and null["num_trades"] > 1 else float("nan"))
    blocks = paired_date_blocks(model["entry_dates"], null["entry_dates"])
    t = comparison["edge_in_standard_errors"]
    return {
        "model_trades": model["num_trades"], "model_mean": model["avg_profit"],
        "model_win_rate": model["win_rate"],
        "model_entry_dates": int(len(np.unique(model["entry_dates"]))),
        "null_trades": null["num_trades"], "null_mean": null["avg_profit"],
        "null_win_rate": null["win_rate"],
        "edge": comparison["edge"], "block_standard_error": comparison["standard_error"],
        "naive_standard_error": naive, "t": t,
        "usable_blocks": sum(1 for a, b in blocks if len(a) and len(b)),
        "verdict": comparison["verdict"],
        "p_one_sided": float(norm.sf(t)) if np.isfinite(t) else float("nan"),
    }


def bin_block_standard_errors(probabilities, profits, dates, curve, seed=PERMUTATION_SEED):
    """Block-bootstrap standard error of each marginal-curve bin's mean, for the figure.

    marginal_ev_curve's own error treats every trade as independent; drawing it would
    show error bars the rest of this project has already shown to be too narrow.
    """
    blocks = date_blocks(dates)
    if len(blocks) < MIN_BLOCKS_FOR_RESAMPLING:
        return [float("nan")] * len(curve)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(blocks), len(blocks)) for _ in range(BOOTSTRAP_REPLICATES)]
    errors = []
    for band in curve:
        upper_inclusive = band is curve[-1]
        inside = (probabilities >= band["lower"]) & (
            (probabilities <= band["upper"]) if upper_inclusive else (probabilities < band["upper"]))
        means = []
        for picks in draws:
            rows = np.concatenate([blocks[i] for i in picks])
            chosen = rows[inside[rows]]
            if len(chosen):
                means.append(profits[chosen].mean())
        errors.append(float(np.std(means, ddof=1)) if len(means) > 1 else float("nan"))
    return errors


def reproduction_gap(category, fitted_scored, validation_frames):
    """Largest absolute difference between the refit model's validation probabilities
    and the saved models/<category> model's. 0 means the refit reproduced it exactly."""
    try:
        saved = SwingTradeDetector(category, AlphaVantageProvider(api_key=None))
    except Exception:
        return float("nan")
    if not saved.is_ready:
        return float("nan")
    saved_scored = score_once(saved, validation_frames)
    gaps = [np.max(np.abs(fitted_scored[s].window_probabilities - saved_scored[s].window_probabilities))
            for s in fitted_scored if s in saved_scored
            and len(fitted_scored[s].window_probabilities) == len(saved_scored[s].window_probabilities)]
    return float(max(gaps)) if gaps else float("nan")


def evaluate_category(category, splits, out, folds):
    swing_threshold = CALIBRATED_SWING_THRESHOLDS.get(category, DEFAULT_SWING_THRESHOLD)
    started = time.time()
    trainer = SwingTradeTrainer(swing_threshold=swing_threshold)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        trainer.train(category_train_dir(category))
    with open(os.path.join(out, "logs", f"{category}_train.txt"), "w", encoding="utf-8") as handle:
        handle.write(log.getvalue())
    stats = trainer.training_stats
    detector = walk_forward_cv.FoldDetector(category, trainer, DEFAULT_DECISION_THRESHOLD)

    # Floor search -- validation/ only, and only on the raw CSVs, symbol column included,
    # exactly as scripts/expected_value_thresholds.run_category reads them.
    validation_raw = load_symbol_data(category_validation_dir(category))
    validation_scored = score_once(detector, validation_raw)
    details = {}
    if stats["is_calibrated"]:
        floor, note, curve = solve_threshold(detector, validation_scored, details)
    else:
        floor, curve = None, []
        note = (f"not calibrated: {stats['calibration_positives']} positives in the calibration "
                f"slice, too few for its scores to be probabilities -- no marginal-return curve "
                f"can be read")
    probabilities, profits, dates = null_trades(detector, validation_scored)
    bin_errors = bin_block_standard_errors(probabilities, profits, dates, curve) if curve else []
    pd.DataFrame({"entry_date": dates, "entry_probability": probabilities, "profit_pct": profits}).to_csv(
        os.path.join(out, "trades", f"{category}_validation_all_entries.csv"), index=False)

    test_probabilities, test_labels = labeled_rows(detector, splits["test"], swing_threshold)
    test_classification = classification(test_probabilities, test_labels)

    result = {
        "category": category,
        "swing_threshold": swing_threshold,
        "train_rows_fit": stats["training_samples"],
        "calibrated": stats["is_calibrated"],
        "calibration_method": stats["calibration_method"],
        "calibration_positives": stats["calibration_positives"],
        "internal_ece": stats["expected_calibration_error"],
        "internal_pr_auc": stats["validation_pr_auc"],
        "internal_base_rate": stats["validation_base_rate"],
        "reproduction_gap": reproduction_gap(category, validation_scored, validation_raw),
        "floor": floor,
        "floor_in_config": CALIBRATED_DECISION_THRESHOLDS.get(category),
        "floor_note": note,
        "floor_details": details,
        "curve": [{**band, "block_standard_error": error} for band, error in zip(curve, bin_errors)],
        "test_classification": test_classification,
        "test": None,
        "test_at_config_floor": None,
        "cross_validation": None,
    }

    if floor is not None:
        test_scored = score_once(detector, load_symbol_data(category_test_dir(category)))
        null_trade_list = trades_at(detector, test_scored, 0.0)
        pd.DataFrame(null_trade_list).to_csv(os.path.join(out, "trades", f"{category}_test_null.csv"), index=False)
        result["test"] = test_against_null(detector, test_scored, floor, null_trade_list,
                                           os.path.join(out, "trades", f"{category}_test_model.csv"))
        # The floor app/config.py ships is written out by hand, rounded. Where the model's
        # calibrated probabilities sit on plateaus -- isotonic calibration produces exactly
        # that -- rounding can move the floor across a plateau and change which trades it
        # admits, which is a different rule from the one validation chose. Reported beside
        # the derived rule so the difference is visible, never in place of it.
        config_floor = CALIBRATED_DECISION_THRESHOLDS.get(category)
        if config_floor is not None:
            low, high = min(floor, config_floor), max(floor, config_floor)
            # Only a difference that changes which bars qualify is a different rule. A floor
            # rounded in its seventh digit, with no probability between the two values in
            # either window, admits exactly the same trades.
            validation_between = int(np.sum((probabilities >= low) & (probabilities < high)))
            test_between = int(sum(np.sum((s.window_probabilities >= low) & (s.window_probabilities < high))
                                   for s in test_scored.values()))
            if validation_between or test_between:
                result["test_at_config_floor"] = {
                    "floor": config_floor,
                    **test_against_null(detector, test_scored, config_floor, null_trade_list,
                                        os.path.join(out, "trades", f"{category}_test_model_config_floor.csv")),
                    "validation_entries_between": validation_between,
                    "test_bars_between": test_between,
                }

    if folds:
        with contextlib.redirect_stdout(io.StringIO()):
            cv = walk_forward_cv.run_category(category, folds)
        result["cross_validation"] = [{
            "fold": f["fold"], "train_end": str(pd.Timestamp(f["train_end"]).date()),
            "validation_end": str(pd.Timestamp(f["validation_end"]).date()),
            "base_rate": f.get("base_rate"), "pr_auc": f.get("pr_auc"), "roc_auc": f.get("roc_auc"),
        } for f in cv["folds"]]

    result["seconds"] = round(time.time() - started, 1)
    return result


# --------------------------------------------------------------------------- output


def _fmt(value, kind):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    formats = {"pct": "{:.2%}", "pct3": "{:.3%}", "spct": "{:+.2%}", "f2": "{:.2f}",
               "sf2": "{:+.2f}", "f3": "{:.3f}", "f4": "{:.4f}", "e": "{:.1e}", "i": "{:.0f}"}
    return formats[kind].format(value) if kind in formats else str(value)


class Tables:
    """Collects tables, writing each as CSV (raw values) and all of them as one Markdown
    file (formatted, with the notes a reader needs)."""

    def __init__(self, out):
        self.out = out
        self.sections = []

    def add(self, key, title, frame, formats, notes):
        frame.to_csv(os.path.join(self.out, "tables", f"{key}.csv"), index=False)
        header = "| " + " | ".join(frame.columns) + " |"
        rule = "|" + "|".join("---" for _ in frame.columns) + "|"
        rows = ["| " + " | ".join(_fmt(row[c], formats.get(c, "")) for c in frame.columns) + " |"
                for _, row in frame.iterrows()]
        body = "\n".join([f"## {key}: {title}", "", header, rule, *rows, ""]
                         + [f"- {n}" for n in notes] + [""])
        self.sections.append(body)

    def write(self, preamble):
        with open(os.path.join(self.out, "tables.md"), "w", encoding="utf-8") as handle:
            handle.write(preamble + "\n\n" + "\n".join(self.sections))


def _style():
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False, "pdf.fonttype": 42,
    })


def _save(fig, out, name):
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _grid(count, columns=4, height=2.2):
    columns = min(columns, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.8 * columns + 0.6, height * rows + 0.6), squeeze=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, axes.flat


def figure_marginal_curves(results, out):
    shown = [r for r in results if r["curve"]]
    if not shown:
        return
    fig, axes = _grid(len(shown))
    for ax, r in zip(axes, shown):
        curve = r["curve"]
        x = np.arange(len(curve))
        means = np.array([b["mean_profit"] for b in curve]) * 100
        errors = np.array([b["block_standard_error"] for b in curve]) * 100
        ax.axhline(0, color=RULE, linewidth=0.8)
        ax.errorbar(x, means, yerr=errors, fmt="o", color=BLUE, markersize=4, elinewidth=1.2, capsize=0)
        if r["floor"] is not None:
            position = next(i for i, b in enumerate(curve) if b["lower"] >= r["floor"])
            ax.axvline(position - 0.5, color=ORANGE, linestyle="--", linewidth=1.2)
        ax.set_title(r["category"].replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b['upper']:.1%}" for b in curve], rotation=90, fontsize=6.5)
        ax.grid(axis="x", visible=False)
    fig.supxlabel("Entry-probability bin (upper edge), validation split", fontsize=8.5)
    fig.supylabel("Mean return per trade (%)", fontsize=8.5)
    fig.suptitle("Marginal return by predicted probability, ±1 block-bootstrap s.e.\n"
                 "(dashed: floor derived on validation)", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig1_marginal_return_curves")


def figure_standard_error_inflation(results, out):
    rows = []
    for r in results:
        best = (r["floor_details"] or {}).get("best")
        if best and best["naive_standard_error"]:
            rows.append((r["category"], "validation: floor separation",
                         best["block_standard_error"] / best["naive_standard_error"]))
        if r["test"] and r["test"]["naive_standard_error"] and np.isfinite(r["test"]["block_standard_error"]):
            rows.append((r["category"], "test: edge over null",
                         r["test"]["block_standard_error"] / r["test"]["naive_standard_error"]))
    if not rows:
        return
    categories = sorted({c for c, _, _ in rows})
    fig, ax = plt.subplots(figsize=(5.2, max(2.6, 0.32 * len(categories) + 1.6)))
    for label, color, marker in (("validation: floor separation", BLUE, "o"), ("test: edge over null", ORANGE, "s")):
        points = [(categories.index(c), v) for c, kind, v in rows if kind == label]
        if points:
            ax.scatter([v for _, v in points], [i for i, _ in points], color=color, marker=marker, s=30,
                       label=label, zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(1, color=RULE, linewidth=1)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace("_", " ") for c in categories])
    ax.set_ylim(-0.6, len(categories) - 0.4)
    ax.set_xlabel("Block-bootstrap s.e. ÷ s.e. assuming independent trades")
    ax.set_xlim(0, max(v for _, _, v in rows) * 1.15)
    fig.legend(loc="lower center", fontsize=7.5, ncol=2)
    ax.set_title("How much treating trades as independent understates the error")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, out, "fig2_standard_error_inflation")


def figure_evidence_against_bars(results, out, bonferroni_bar):
    searched = [r for r in results if (r["floor_details"] or {}).get("best")]
    tested = [r for r in results if r["test"] and np.isfinite(r["test"]["t"])]
    if not searched:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, max(2.9, 0.3 * len(searched) + 2.0)),
                                      gridspec_kw={"width_ratios": [3, 2]})
    y = np.arange(len(searched))
    left.scatter([r["floor_details"]["best"]["t"] for r in searched], y, color=BLUE, s=30, zorder=3,
                 label="best candidate floor")
    left.scatter([r["floor_details"]["required_t"] for r in searched], y, color=INK, marker="|", s=160,
                 zorder=3, label="bar after correcting for the search")
    left.set_yticks(y)
    left.set_yticklabels([r["category"].replace("_", " ") for r in searched])
    left.set_ylim(-0.6, len(searched) - 0.4)
    left.set_xlabel("Separation (block-bootstrap s.e.)")
    left.set_title("Validation: floor search")

    if tested:
        yt = np.arange(len(tested))
        right.scatter([r["test"]["t"] for r in tested], yt, color=ORANGE, marker="s", s=30, zorder=3,
                      label="edge at the derived floor")
        right.axvline(PROJECT_BAR, color=INK, linestyle="--", linewidth=1, label="project bar (2 s.e.)")
        right.axvline(bonferroni_bar, color=INK_MUTED, linestyle=":", linewidth=1.2,
                      label=f"Bonferroni, 8 categories ({bonferroni_bar:.2f})")
        right.set_yticks(yt)
        right.set_yticklabels([r["category"].replace("_", " ") for r in tested])
        right.set_ylim(-0.6, len(tested) - 0.4)
        right.set_xlim(min(0, min(r["test"]["t"] for r in tested) - 0.5),
                       max(bonferroni_bar, max(r["test"]["t"] for r in tested)) + 0.5)

    else:
        right.text(0.5, 0.5, "no category produced a floor\nto test", ha="center", va="center",
                   transform=right.transAxes, color=INK_MUTED)
        right.set_yticks([])
    right.set_xlabel("Edge over null (s.e.)")
    right.set_title("Test: at the validation floor")
    fig.legend(loc="lower center", fontsize=7, ncol=3)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _save(fig, out, "fig3_evidence_against_bars")


def figure_horizons(horizon_frame, out):
    categories = list(dict.fromkeys(horizon_frame["category"]))
    # Dots, not bars: the values span two orders of magnitude and need a log axis, where a
    # bar's length measures nothing.
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(categories))
    for (label, _, _), color, marker in zip(HORIZONS, (BLUE, ORANGE, AQUA), ("o", "s", "D")):
        values = [horizon_frame[(horizon_frame.category == c) & (horizon_frame.horizon == label)]
                  ["effective_independent_outcomes"].iloc[0] for c in categories]
        ax.scatter(x, values, color=color, marker=marker, s=34, label=label, zorder=3,
                   edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=7)
    ax.set_ylabel("Independent outcomes (log scale)")
    ax.set_title("Non-overlapping holding periods × effective independent series, whole history")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    _save(fig, out, "fig4_independent_outcomes_by_horizon")


def figure_leakage(universe_frame, out):
    frame = universe_frame.sort_values("leak_per_symbol_split")
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(frame) + 1.2))
    y = np.arange(len(frame))
    ax.barh(y, frame["leak_per_symbol_split"] * 100, color=BLUE, height=0.6)
    for i, value in enumerate(frame["leak_per_symbol_split"]):
        ax.text(value * 100 + 1, i, f"{value:.0%}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in frame["category"]])
    ax.set_xlim(0, max(10, frame["leak_per_symbol_split"].max() * 100 * 1.18))
    ax.set_xlabel("Test rows dated on a day a sibling ticker trained on (%)")
    calendar = frame["leak_calendar_split"].max()
    ax.set_title(f"Leakage under a per-symbol percentage split\n"
                 f"(category-wide calendar split used now: {calendar:.0%} in every category)", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig5_split_leakage")


def figure_reliability(results, out):
    shown = [r for r in results if r["calibrated"] and r["test_classification"]["reliability"]]
    if not shown:
        return
    fig, axes = _grid(len(shown), height=2.1)
    for ax, r in zip(axes, shown):
        bins = r["test_classification"]["reliability"]
        predicted = [b["predicted"] * 100 for b in bins]
        observed = [b["observed"] * 100 for b in bins]
        top = max(max(predicted), max(observed), 0.1) * 1.08
        ax.plot([0, top], [0, top], color=RULE, linewidth=1)
        ax.plot(predicted, observed, "-o", color=BLUE, markersize=3.5, linewidth=1.5)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(f"{r['category'].replace('_', ' ')}\nECE {r['test_classification']['ece']:.3f}", fontsize=8)
    fig.supxlabel("Mean predicted probability (%), test split, decile bins", fontsize=8.5)
    fig.supylabel("Observed positive rate (%)", fontsize=8.5)
    fig.suptitle("Calibration out of sample (diagonal = perfectly calibrated)", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig6_test_reliability")


LABEL_OFFSETS = {
    "energy_commodity": (-5, 6, "right"), "credit_conditions": (-6, -3, "right"),
    "market_beta": (0, -12, "center"), "growth_tech": (6, -3, "left"),
    "international_emerging": (6, -3, "left"),
}


def figure_effective_series(universe_frame, out):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    top = universe_frame["symbols"].max() + 9  # room for labels right of the widest categories
    ax.plot([0, top], [0, top], color=RULE, linewidth=1)
    ax.scatter(universe_frame["symbols"], universe_frame["effective_series"], color=BLUE, s=30, zorder=3)
    for _, row in universe_frame.iterrows():
        # Several categories sit close together; each label is placed by hand so none
        # overlaps a neighbour.
        dx, dy, ha = LABEL_OFFSETS.get(row["category"], (5, 4, "left"))
        ax.annotate(row["category"].replace("_", " "), (row["symbols"], row["effective_series"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=6.5, color=INK_MUTED, ha=ha)
    ax.set_xlim(0, top)
    ax.set_ylim(0, max(6, universe_frame["effective_series"].max() + 1))
    ax.set_xlabel("Tickers in category")
    ax.set_ylabel("Effective independent series")
    ax.set_title("Many tickers, few independent series\n(grey line: every ticker independent)")
    fig.tight_layout()
    _save(fig, out, "fig7_effective_series")


# ----------------------------------------------------------------------------- main


def _data_measurements(categories, splits):
    universe_rows, label_rows, horizon_rows = [], [], []
    for category in categories:
        s = splits[category]
        n_eff, mean_correlation = effective_series(s)
        leak_old, leak_new = leakage(s)
        sessions, common_start = coverage_sessions(s)
        swing = CALIBRATED_SWING_THRESHOLDS.get(category, DEFAULT_SWING_THRESHOLD)
        row = {"category": category, "tickers_configured": len(FACTOR_CATEGORIES[category]),
               "symbols": len(full_histories(s)), "common_start": str(common_start.date()),
               "coverage_sessions": sessions}
        for name in ("train", "validation", "test"):
            frames = s[name]
            row[f"{name}_rows"] = sum(len(f) for f in frames.values())
            row[f"{name}_first"] = str(min(f.index.min() for f in frames.values()).date()) if frames else None
            row[f"{name}_last"] = str(max(f.index.max() for f in frames.values()).date()) if frames else None
        row.update({"mean_pairwise_correlation": mean_correlation, "effective_series": n_eff,
                    "leak_per_symbol_split": leak_old, "leak_calendar_split": leak_new})
        universe_rows.append(row)

        label_specs = [(label, swing, m, lf) for label, m, lf in HORIZONS[1:]]
        if category == "growth_tech":
            # Checks the figure app/labeling.py used to quote: +100% within 126-252 sessions.
            label_specs.append((HORIZONS[2][0] + ", +100% bar", 1.0, 126, 252))
        for label, threshold, min_hold, lookforward in label_specs:
            rates = label_rates(s["train"], threshold, min_hold, lookforward)
            label_rows.append({"category": category, "horizon": label, "swing_threshold": threshold,
                               "terminal_rate": rates["terminal"], "peak_rate": rates["peak"],
                               "peak_over_terminal": (rates["peak"] / rates["terminal"]
                                                      if rates["terminal"] else float("nan"))})

        for label, _, lookforward in HORIZONS:
            need = sum(_required_sessions(lookforward))
            windows = sessions // lookforward
            horizon_rows.append({"category": category, "horizon": label, "sessions_needed": need,
                                 "coverage_sessions": sessions, "fits_three_way_split": sessions >= need,
                                 "non_overlapping_holds": windows, "effective_series": n_eff,
                                 "effective_independent_outcomes": windows * n_eff})
        print(f"  {category}: data measurements done", flush=True)
    return universe_rows, label_rows, horizon_rows


def _write_tables(out, run, universe, labels_frame, horizons, results, folds, bonferroni_bar):
    family = len(list_categories())
    tables = Tables(out)
    tables.add("T1", "Universe and splits", universe[[
        "category", "tickers_configured", "symbols", "train_rows", "train_first", "train_last",
        "validation_rows", "validation_first", "validation_last", "test_rows", "test_first",
        "test_last"]], {}, [
        "Rows are symbol-days summed across the category. Train, validation and test share one "
        "pair of calendar cutoffs per category, with a 126-row embargo dropped at each seam.",
        "`symbols` below `tickers_configured` means a configured ticker has no file in any split.",
    ])
    tables.add("T2", "How many independent series a category really holds", universe[[
        "category", "symbols", "mean_pairwise_correlation", "effective_series"]],
        {"mean_pairwise_correlation": "f2", "effective_series": "f2"}, [
        "Effective series = n / (1 + (n - 1) * mean pairwise correlation of daily returns), over "
        "all three splits, with returns taken within each split file so none spans an embargo gap.",
    ])
    tables.add("T3", "Leakage from splitting each symbol separately", universe[[
        "category", "leak_per_symbol_split", "leak_calendar_split"]],
        {"leak_per_symbol_split": "pct", "leak_calendar_split": "pct"}, [
        "Share of test rows dated on a calendar day that some other symbol in the same category "
        "was trained on. Per-symbol split: oldest 55% of each symbol's own rows to train, newest "
        "30% to test (the split this project used before). Calendar split: the one on disk.",
        "Measured on the current universe and data, not the older universe the original "
        "39-49% figure came from; each symbol's history is the three files joined, so it lacks "
        "the two embargoed seams.",
    ])
    tables.add("T4", "Label definition: terminal (median close) vs peak (maximum high)", labels_frame,
               {"terminal_rate": "pct", "peak_rate": "pct", "swing_threshold": "pct",
                "peak_over_terminal": "f2"}, [
        "Positive-label rate on train/, rows with a complete outcome window only, at each "
        "category's configured swing threshold (3% floor applies).",
        "The current models use terminal labels at 63-126 sessions.",
    ])
    tables.add("T5", "Evidence available at each holding horizon", horizons,
               {"effective_series": "f2", "effective_independent_outcomes": "f2"}, [
        "coverage_sessions counts trading days from the category's common start (its median "
        "symbol's first date) to the end of the data, including the two embargoed seams.",
        "sessions_needed is the project's own split-sizing rule (scripts/build_factor_datasets."
        "_required_sessions) evaluated at each horizon: train + validation + test.",
        "effective_independent_outcomes = non_overlapping_holds x effective_series: roughly how "
        "many genuinely separate outcomes the whole history offers, before any split.",
    ])

    tables.add("T6", "Models: calibration and ranking quality out of sample", pd.DataFrame([{
        "category": r["category"], "swing_threshold": r["swing_threshold"],
        "fit_rows": r["train_rows_fit"], "calibration": r["calibration_method"] or "none",
        "calibration_positives": r["calibration_positives"], "internal_ece": r["internal_ece"],
        "test_rows": r["test_classification"]["rows"],
        "test_base_rate": r["test_classification"]["base_rate"],
        "test_pr_auc": r["test_classification"]["pr_auc"],
        "test_pr_auc_lift": r["test_classification"]["pr_auc_lift"],
        "test_roc_auc": r["test_classification"]["roc_auc"],
        "test_ece": r["test_classification"]["ece"],
        "max_gap_vs_saved_model": r["reproduction_gap"],
    } for r in results]), {
        "swing_threshold": "pct", "internal_ece": "f4", "test_base_rate": "pct", "test_pr_auc": "f4",
        "test_pr_auc_lift": "f2", "test_roc_auc": "f3", "test_ece": "f4",
        "max_gap_vs_saved_model": "e"}, [
        "Test rows are every scorable row with a complete 126-session outcome window. PR-AUC "
        "lift is PR-AUC over the base rate (1.0 = random ranking); ROC-AUC 0.5 = random.",
        "These rows overlap by up to 99% of their outcome window, so no standard error is "
        "attached: these AUCs describe one path through history, not a sample of many.",
        "max_gap_vs_saved_model: largest difference in any validation probability between this "
        "run's refit and the model saved in models/. 0 means the refit reproduced it exactly.",
    ])

    floor_rows = []
    for r in results:
        d = r["floor_details"] or {}
        best = d.get("best") or {}
        floor_rows.append({
            "category": r["category"], "validation_trades": d.get("num_trades"),
            "entry_dates": d.get("num_entry_dates"), "date_blocks": d.get("num_blocks"),
            "candidates": len(d.get("candidates", [])), "best_floor": best.get("threshold"),
            "separation": best.get("separation"), "block_se": best.get("block_standard_error"),
            "naive_se": best.get("naive_standard_error"), "t": best.get("t"),
            "bar": d.get("required_t"), "floor": r["floor"], "floor_in_config": r["floor_in_config"],
        })
    tables.add("T7", "Floor search on validation (scripts/expected_value_thresholds.py)",
               pd.DataFrame(floor_rows), {
        "validation_trades": "i", "entry_dates": "i", "date_blocks": "i",
        "best_floor": "pct3", "separation": "spct", "block_se": "pct", "naive_se": "pct",
        "t": "f2", "bar": "f2", "floor": "pct3", "floor_in_config": "pct3"}, [
        "Every trade the model would open with no threshold, binned by predicted probability. A "
        "floor ships only if trades above it out-earn trades below it by more than the 95th "
        "percentile of the best-of-candidates statistic under a block permutation (`bar`).",
        f"Blocks are {BLOCK_LENGTH_CALENDAR_DAYS} calendar days (one holding period); fewer than "
        f"{MIN_BLOCKS_FOR_RESAMPLING} and no standard error is reported.",
        "naive_se is the same separation's error if every trade were independent -- shown for "
        "comparison only.",
        "floor_in_config is the value app/config.py carries; a match means this run re-derived it.",
    ] + [f"{r['category']}: {r['floor_note']}" for r in results])

    def test_row(label, floor, test):
        return {
            "category": label, "floor": floor, "model_trades": test["model_trades"],
            "model_entry_dates": test["model_entry_dates"], "model_mean": test["model_mean"],
            "null_trades": test["null_trades"], "null_mean": test["null_mean"],
            "edge": test["edge"], "block_se": test["block_standard_error"],
            "naive_se": test["naive_standard_error"], "usable_blocks": test["usable_blocks"],
            "t": test["t"], "p_one_sided": test["p_one_sided"],
            "p_bonferroni_8": (min(1.0, test["p_one_sided"] * 2 * family)
                               if np.isfinite(test["p_one_sided"]) else float("nan")),
            "verdict": test["verdict"],
        }

    test_rows, disclosures = [], []
    for r in results:
        if r["test"]:
            test_rows.append(test_row(r["category"], r["floor"], r["test"]))
        if r["test_at_config_floor"]:
            shipped = r["test_at_config_floor"]
            test_rows.append(test_row(f"{r['category']} (config's rounded floor, not the derived rule)",
                                      shipped["floor"], shipped))
            disclosures.append(
                f"{r['category']}: app/config.py ships {shipped['floor']!r} where validation derived "
                f"{r['floor']!r}. {shipped['validation_entries_between']} of the "
                f"{r['floor_details'].get('num_trades')} validation entries (and "
                f"{shipped['test_bars_between']} test bars) lie between the two -- calibrated "
                f"probabilities sit on plateaus, and rounding moved the floor across one -- so the "
                f"shipped value is a different rule that validation never chose. The derived row is "
                f"the result; the other is shown because earlier write-ups quoted it.")
    tables.add("T8", "Out of sample: model at its validation floor vs the model-off null",
               pd.DataFrame(test_rows) if test_rows else
               pd.DataFrame([{"category": "none", "verdict": "no category produced a floor"}]), {
        "floor": "pct3", "model_mean": "spct", "null_mean": "spct", "edge": "spct", "block_se": "pct",
        "naive_se": "pct", "t": "sf2", "p_one_sided": "f3", "p_bonferroni_8": "f3"}, [
        "Null: the same entry/exit machinery with the threshold at zero -- every trade the model "
        "would open if its ranking were ignored. Edge = mean return per trade, model minus null.",
        f"Project bar: {PROJECT_BAR:.0f} block-bootstrap standard errors. Descriptive only: the "
        f"two-sided 5% bar after Bonferroni over {family} categories is {bonferroni_bar:.2f} s.e.; "
        f"p_bonferroni_8 is the two-sided p-value times {family} (capped at 1).",
        "Only categories with a validation floor have a rule to test; the rest are in T7.",
    ] + disclosures)

    if folds:
        cv_rows = []
        for r in results:
            fold_rows = r["cross_validation"] or []
            lifts = [f["pr_auc"] / f["base_rate"] for f in fold_rows
                     if f.get("base_rate") and f.get("pr_auc") == f.get("pr_auc")]
            roc = [f["roc_auc"] for f in fold_rows if f.get("roc_auc") == f.get("roc_auc")]
            cv_rows.append({
                "category": r["category"], "folds": len(fold_rows),
                "pr_auc_lift_mean": float(np.mean(lifts)) if lifts else float("nan"),
                "pr_auc_lift_min": float(np.min(lifts)) if lifts else float("nan"),
                "pr_auc_lift_max": float(np.max(lifts)) if lifts else float("nan"),
                "roc_auc_mean": float(np.mean(roc)) if roc else float("nan"),
                "roc_auc_sd": float(np.std(roc, ddof=1)) if len(roc) > 1 else float("nan"),
                "folds_roc_below_half": sum(1 for v in roc if v < 0.5),
            })
        tables.add("T9", f"Walk-forward cross-validation ({folds} expanding folds, pre-test data only)",
                   pd.DataFrame(cv_rows), {
            "pr_auc_lift_mean": "f2", "pr_auc_lift_min": "f2", "pr_auc_lift_max": "f2",
            "roc_auc_mean": "f3", "roc_auc_sd": "f3"}, [
            "scripts/walk_forward_cv.py: each fold trains on everything before its cutoff and is "
            "scored on the next block, embargoed. test/ is never read.",
            "Ranking quality only. Fold-level trading edges are not reported: each fold's block "
            "holds too few non-overlapping holding periods to resample.",
        ])

    stamp = (f"Commit `{run['git_commit']}`" + (" -- **working tree was dirty; not a citable run**"
                                                if run["git_dirty"] else ", clean working tree"))
    tables.write(f"# Paper tables\n\nGenerated by `scripts/paper_run.py` at {run['finished_at']}. "
                 f"{stamp}. Data: `paper/data_manifest.json` (sha256 "
                 f"`{run['data_manifest_sha256'][:16]}...`). Holding period "
                 f"{DEFAULT_MIN_HOLD_PERIODS}-{DEFAULT_LOOKFORWARD_PERIODS} sessions, {LABEL_MODE} "
                 f"labels. Every number below is regenerated by the script; do not edit by hand.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--freeze-data", action="store_true", help="record the dataset hashes and exit")
    parser.add_argument("--require-clean", action="store_true", help="refuse a dirty working tree")
    parser.add_argument("--folds", type=int, default=walk_forward_cv.DEFAULT_FOLDS,
                        help="cross-validation folds per category (0 skips)")
    parser.add_argument("--categories", help="comma-separated subset, for trial runs")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.freeze_data:
        freeze_data()
        return 0

    run = provenance()
    if args.require_clean and run["git_dirty"]:
        print("Refusing: the working tree has uncommitted changes, so this run could not be "
              "tied to a commit:\n  " + "\n  ".join(run["git_changed_files"]))
        return 2
    data_manifest = verify_data()
    run["data_manifest_sha256"] = _sha256(DATA_MANIFEST_PATH)
    run["data_cutoffs"] = data_manifest["cutoffs"]

    categories = args.categories.split(",") if args.categories else list_categories()
    out = os.path.abspath(args.out)
    for sub in ("tables", "figures", "trades", "logs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    _style()
    print(f"Data verified ({len(data_manifest['files'])} files). Commit {run['git_commit']}"
          f"{' (DIRTY working tree -- not a citable run)' if run['git_dirty'] else ''}.", flush=True)

    splits = {c: split_frames(c) for c in categories}
    universe_rows, label_rows, horizon_rows = _data_measurements(categories, splits)
    universe, labels_frame, horizons = (pd.DataFrame(universe_rows), pd.DataFrame(label_rows),
                                        pd.DataFrame(horizon_rows))

    results = []
    for category in categories:
        print(f"  {category}: training and evaluating...", flush=True)
        result = evaluate_category(category, splits[category], out, args.folds)
        results.append(result)
        verdict = (f"floor {result['floor']:.4%}, test edge {result['test']['edge']:+.2%} "
                   f"at {result['test']['t']:+.2f} s.e." if result["test"] else "no floor")
        print(f"    done in {result['seconds']:.0f}s -- {verdict}", flush=True)

    bonferroni_bar = float(norm.isf(FAMILY_ALPHA / 2 / len(list_categories())))
    run["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run["categories"] = categories
    run["cross_validation_folds"] = args.folds
    run["bonferroni_bar"] = bonferroni_bar

    _write_tables(out, run, universe, labels_frame, horizons, results, args.folds, bonferroni_bar)
    figure_marginal_curves(results, out)
    figure_standard_error_inflation(results, out)
    figure_evidence_against_bars(results, out, bonferroni_bar)
    figure_horizons(horizons, out)
    figure_leakage(universe, out)
    figure_reliability(results, out)
    figure_effective_series(universe, out)

    with open(os.path.join(out, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=1, default=str)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"universe": universe_rows, "labels": label_rows, "horizons": horizon_rows,
                   "categories": results}, handle, indent=1, default=str)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
