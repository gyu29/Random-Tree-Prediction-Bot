"""Trains one SwingTradeTrainer per factor category.

Two distinct splits are in play, and it's worth being explicit about both:

1. Category-level walk-forward split (the point of this whole rewrite): the caller
   points `train()` at train/<category>/ and the resulting model is evaluated
   separately against test/<category>/ (see scripts/train_all_categories.py). Every
   test row for a symbol is chronologically after every train row for that symbol.

2. Internal train/validation split *within* train(): the original implementation used
   sklearn's train_test_split with its default shuffle=True on overlapping rolling-window
   features -- a random shuffle on autocorrelated time-series rows leaks adjacent-window
   information across the split and inflates the reported score. This version instead
   holds out the most recent slice of each symbol's rows, chronologically, purely so
   `train()` has something to report progress against; it is not the out-of-sample
   evaluation that matters (that's #1). A chronological cut alone still leaks, because
   a swing label looks `lookforward_periods` bars into the future: the last rows before
   the cut are labelled from bars that land on the validation side. So the cut carries
   an embargo -- `lookforward_periods` rows immediately before it belong to neither
   side (see `_chronological_internal_split`).

Reported metrics: accuracy is deliberately not the headline. Positive labels run
0.6-8% of rows depending on category, so "always predict no swing" scores 92-99%
and beats every model this trainer has produced. `train()` prints and stores
PR-AUC, ROC-AUC, precision and recall alongside accuracy *and* the majority-class
baseline it has to beat, so a high accuracy can't be mistaken for a working model.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from app.config import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_LOOKFORWARD_PERIODS,
    DEFAULT_MIN_HOLD_PERIODS,
    DEFAULT_SWING_THRESHOLD,
)
from app.data_loader import DataProcessor
from app.ensemble import HybridSwingEnsemble, compute_scale_pos_weight
from app.indicators import PRICE_LEVEL_COLUMNS, TechnicalIndicators
from app.labeling import create_swing_labels, effective_threshold
from app.market_context import MARKET_CONTEXT_FEATURES, load_context

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

INTERNAL_VALIDATION_FRACTION = 0.15
# Slice reserved for mapping each ensemble member's raw scores onto real probabilities
# (app/ensemble.py). It sits between the fitting slice and the reporting slice, purged
# from both, so calibration never sees data it was fitted on and the reported metrics
# never see data the calibration was fitted on.
#
# 20%, not 15%. The binding constraint is the category with the fewest positives in this
# window, not the average one: at 15% credit_conditions had 78, below
# app.ensemble.MIN_POSITIVES_FOR_ISOTONIC, so it fell back to a two-parameter sigmoid
# that cannot represent a distortion of any other shape. 20% is the smallest fraction at
# which every category that calibrates at all reaches the non-parametric fit -- 172 for
# credit_conditions, and more for the rest.
#
# It costs the fit slice little: credit_conditions goes from 2474 positive training
# examples to 2388, about 3.5%. Enlarging this window extends it backwards in time, into
# the earlier period where that category's swings actually cluster, which is why it gains
# more than the proportional share.
INTERNAL_CALIBRATION_FRACTION = 0.20

# Raw OHLCV, bookkeeping columns, and the label columns themselves -- plus every
# absolute price/volume level (app/indicators.py's PRICE_LEVEL_COLUMNS, which owns
# that list and explains why they can't be features). Each excluded level has a
# scale-free counterpart that *is* a feature, so nothing is lost but the part a
# tree could never extrapolate past.
EXCLUDE_FROM_FEATURES = [
    "open", "high", "low", "close", "adj_close", "volume", "dividends",
    "stock_splits", "symbol", "swing_label", "swing_profit_potential", "swing_risk",
] + list(PRICE_LEVEL_COLUMNS)


class SwingTradeTrainer:
    def __init__(
        self,
        swing_threshold=DEFAULT_SWING_THRESHOLD,
        lookforward_periods=DEFAULT_LOOKFORWARD_PERIODS,
        min_hold_periods=DEFAULT_MIN_HOLD_PERIODS,
        decision_threshold=DEFAULT_DECISION_THRESHOLD,
        rf_estimators=250,
        xgb_learning_rate=0.05,
        xgb_max_depth=6,
        market_context=None,
    ):
        if XGBClassifier is None:
            raise ImportError("xgboost is required for training. Install it with: pip install xgboost")
        self.swing_threshold = swing_threshold
        self.lookforward_periods = lookforward_periods
        self.min_hold_periods = min_hold_periods
        self.decision_threshold = decision_threshold
        # Loaded here rather than per symbol: the same four series serve every symbol, and
        # re-reading them once per ticker would dominate feature construction. None means
        # the cache has not been built, and the model is trained without context.
        self.market_context = load_context() if market_context is None else market_context

        self.random_forest_model = RandomForestClassifier(
            n_estimators=rf_estimators, max_depth=10, min_samples_split=20, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=-1,
            class_weight="balanced_subsample", bootstrap=True,
        )
        self.xgboost_model = XGBClassifier(
            n_estimators=300, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate, subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="aucpr", random_state=42, n_jobs=-1,
            min_child_weight=3, reg_lambda=1.0, reg_alpha=0.0, tree_method="hist",
        )
        self.model = HybridSwingEnsemble(self.random_forest_model, self.xgboost_model, decision_threshold=decision_threshold)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.training_stats = {}
        self.scale_pos_weight = None
        self._embargoed_rows = 0

    def load_historical_data(self, data_directory):
        all_files = sorted(
            f for f in os.listdir(data_directory) if f.lower().endswith((".csv", ".parquet", ".xlsx"))
        )
        if not all_files:
            raise ValueError(f"No supported data files found in {data_directory}")

        dataframes = []
        loaded_files = []
        for filename in all_files:
            file_path = os.path.join(data_directory, filename)
            try:
                df = DataProcessor.load_and_validate_data(file_path)
                if len(df) > 0:
                    dataframes.append(df)
                    loaded_files.append(file_path)
            except Exception as error:
                print(f"Error loading {file_path}: {error}")
                continue
        if not dataframes:
            raise ValueError("No valid data files could be loaded")

        combined_df = pd.concat(dataframes, ignore_index=False).sort_values(["symbol"])
        self._loaded_files = loaded_files
        return combined_df

    def prepare_training_data(self, df):
        grouped_frames = []
        for symbol, symbol_df in df.groupby("symbol", sort=True):
            symbol_frame = symbol_df.drop(columns=["symbol"]).sort_index()
            df_features = TechnicalIndicators.create_all_indicators(
                symbol_frame, market_context=self.market_context
            )
            df_labeled = create_swing_labels(
                df_features, self.swing_threshold, self.lookforward_periods, self.min_hold_periods
            )
            # Fill gaps against this symbol's own history, before the concat. Filling
            # afterwards lets one symbol supply another's values: a symbol with too
            # little history for a 200-day window has no sma_200/price_sma_200_ratio
            # column at all, and a frame-wide ffill hands it the neighbouring symbol's
            # numbers instead of leaving it missing. Category-wide calendar cutoffs
            # (scripts/build_factor_datasets.py) make short per-symbol training frames
            # routine, so this is a live case, not a theoretical one.
            #
            # Forward only, never backward. A back-fill takes a value from later in the
            # series and writes it into an earlier row, which is look-ahead however
            # short the reach: for a 200-day moving average it silently seeded the first
            # 200 rows with a number computed from bars they precede, and once market
            # context joined the frame it would have handed a 1950s bar the VIX reading
            # from 1990. Leading rows with nothing to carry forward stay NaN and are
            # dropped below, which costs a warm-up window per symbol and no correctness.
            fill_cols = [col for col in df_labeled.columns if col not in EXCLUDE_FROM_FEATURES]
            df_labeled[fill_cols] = df_labeled[fill_cols].ffill()
            df_labeled["symbol"] = symbol
            grouped_frames.append(df_labeled)

        # kind="stable": _chronological_internal_split cuts each symbol's rows by
        # position, so within-symbol chronological order has to survive this sort.
        # The default quicksort makes no such guarantee for equal keys.
        df_labeled = pd.concat(grouped_frames, ignore_index=False).sort_values(["symbol"], kind="stable")

        feature_cols = [col for col in df_labeled.columns if col not in EXCLUDE_FROM_FEATURES]
        df_labeled[feature_cols] = df_labeled[feature_cols].replace([np.inf, -np.inf], 0)
        df_clean = df_labeled.dropna(subset=feature_cols + ["swing_label"])

        # A symbol short enough to be missing a whole feature column loses every row
        # here. That's the right outcome, but say so rather than shrinking the training
        # set silently.
        dropped_symbols = sorted(set(df_labeled["symbol"]) - set(df_clean["symbol"]))
        if dropped_symbols:
            print(f"WARNING: {', '.join(dropped_symbols)} contributed no usable rows -- too "
                  f"little history to compute every feature. Excluded from training.")

        self.feature_columns = feature_cols
        return df_clean[feature_cols], df_clean["swing_label"], df_clean

    def _chronological_internal_split(self, df_clean, X, y):
        """Per-symbol chronological three-way split with an embargo at each boundary.

        Each symbol's rows, oldest to newest:

            [ fit ][embargo][ calibrate ][embargo][ report ]

        `fit` trains the trees, `calibrate` maps their raw scores onto probabilities, and
        `report` is what train() prints metrics on -- three disjoint slices, because
        reusing one for two purposes makes the reported number optimistic by exactly the
        amount that matters. See the module docstring for why this isn't a random shuffle,
        and why a bare chronological cut still isn't enough.

        The `lookforward_periods` rows before each boundary belong to no slice: their
        swing labels are computed from bars on the far side of it.

        Masks are positional, not index-label based. df_clean's index is the date index
        and symbols share dates, so `mask.loc[one_symbol_slice.index] = False` would also
        flip every other symbol's rows on those dates.
        """
        symbols = df_clean["symbol"].to_numpy()
        fit_mask = np.zeros(len(df_clean), dtype=bool)
        calibration_mask = np.zeros(len(df_clean), dtype=bool)
        report_mask = np.zeros(len(df_clean), dtype=bool)
        embargoed = 0

        for symbol in np.unique(symbols):
            positions = np.flatnonzero(symbols == symbol)
            rows = len(positions)
            report_start = int(rows * (1 - INTERNAL_VALIDATION_FRACTION))
            calibration_end = max(0, report_start - self.lookforward_periods)
            calibration_start = int(rows * (1 - INTERNAL_VALIDATION_FRACTION - INTERNAL_CALIBRATION_FRACTION))
            fit_end = max(0, calibration_start - self.lookforward_periods)

            report_mask[positions[report_start:]] = True
            # A symbol too short to leave a usable calibration slice contributes none --
            # better than a two-row calibration set that produces a nonsense mapping.
            if calibration_start < calibration_end:
                calibration_mask[positions[calibration_start:calibration_end]] = True
                fit_mask[positions[:fit_end]] = True
                embargoed += (calibration_start - fit_end) + (report_start - calibration_end)
            else:
                fit_mask[positions[:calibration_end]] = True
                embargoed += report_start - calibration_end

        self._embargoed_rows = embargoed
        return {
            "fit": (X[fit_mask], y[fit_mask]),
            "calibrate": (X[calibration_mask], y[calibration_mask]),
            "report": (X[report_mask], y[report_mask]),
        }

    def train(self, data_directory):
        df = self.load_historical_data(data_directory)
        if len(df) < 500:
            raise ValueError("Insufficient data for training (minimum 500 samples required)")

        X, y, df_clean = self.prepare_training_data(df)
        context_columns = [c for c in MARKET_CONTEXT_FEATURES if c in self.feature_columns]
        print(f"Training dataset shape: {X.shape} "
              f"({len(context_columns)} market-context features"
              f"{'' if context_columns else ' -- context cache not built'})")
        print(f"Swing opportunities: {sum(y)} out of {len(y)} samples ({sum(y) / len(y) * 100:.2f}%)")
        if sum(y) < 10:
            print("WARNING: very few positive samples -- consider a lower swing threshold or more data.")

        slices = self._chronological_internal_split(df_clean, X, y)
        X_train, y_train = slices["fit"]
        X_calibrate, y_calibrate = slices["calibrate"]
        X_val, y_val = slices["report"]
        print(f"Internal split: {len(X_train)} fit rows, {len(X_calibrate)} calibration rows, "
              f"{len(X_val)} report rows, {self._embargoed_rows} embargoed "
              f"({self.lookforward_periods} per boundary per symbol)")

        # Fitted on the fit slice alone: a scaler fitted across the calibration or report
        # slices would carry their distribution into the model's inputs.
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_calibrate_scaled = self.scaler.transform(X_calibrate) if len(X_calibrate) else X_calibrate
        X_val_scaled = self.scaler.transform(X_val)

        self.scale_pos_weight = compute_scale_pos_weight(y_train)
        self.xgboost_model.set_params(scale_pos_weight=self.scale_pos_weight)
        print(f"Computed scale_pos_weight={self.scale_pos_weight:.3f} from actual training label ratio")

        if len(X_calibrate):
            self.model.fit_calibrated(X_train_scaled, y_train, X_calibrate_scaled, y_calibrate)
        else:
            self.model.fit(X_train_scaled, y_train)
        if self.model.is_calibrated:
            print(f"Calibrated both ensemble members with {self.model.calibration_method} on "
                  f"{len(X_calibrate)} held-out rows ({int(np.sum(y_calibrate))} positives)")
        else:
            print("WARNING: not calibrated -- the calibration slice had no positive labels. "
                  "Probabilities are raw scores and are not comparable across categories.")
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        y_val_proba = self.model.predict_proba(X_val_scaled)

        train_score = accuracy_score(y_train, y_train_pred)
        val_score = accuracy_score(y_val, y_val_pred)
        metrics = self._validation_metrics(y_val, y_val_pred, y_val_proba[:, 1])
        calibration_error = self._expected_calibration_error(y_val, y_val_proba[:, 1])
        print(f"Expected calibration error on the report slice: {calibration_error:.4f} "
              f"(0 = predicted probabilities match observed frequencies)")

        # PR-AUC first, and accuracy only next to the baseline it has to beat: with a
        # ~1-8% positive rate, "always predict no swing" scores in the 90s and beats
        # every model trained here, so a bare accuracy reads as success when it isn't.
        if metrics["base_rate"] > 0:
            print(f"Validation PR-AUC: {metrics['pr_auc']:.4f}  (base rate {metrics['base_rate']:.2%}, "
                  f"which is what a random ranker scores -- "
                  f"{metrics['pr_auc'] / metrics['base_rate']:.1f}x lift)")
        else:
            print("Validation PR-AUC: n/a (no positive labels in the validation slice)")
        print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}  |  "
              f"precision: {metrics['precision']:.4f}  |  recall: {metrics['recall']:.4f}  "
              f"(at decision_threshold={self.decision_threshold:.2f})")
        print(f"Validation accuracy: {val_score:.4f} vs. {metrics['majority_accuracy']:.4f} for always-negative "
              f"-- {'BEATS' if val_score > metrics['majority_accuracy'] else 'does NOT beat'} the baseline. "
              f"Train accuracy: {train_score:.4f}")
        print(classification_report(y_val, y_val_pred, zero_division=0))
        print(confusion_matrix(y_val, y_val_pred))

        feature_importance = pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        self.training_stats = {
            "train_score": float(train_score),
            "validation_score": float(val_score),
            "test_score": float(val_score),  # kept for backward-compatible readers
            # The metrics that survive a 1-8% positive rate. validation_score above is
            # accuracy, kept for existing readers (ui/, scripts/) but not the headline.
            "validation_pr_auc": metrics["pr_auc"],
            "validation_roc_auc": metrics["roc_auc"],
            "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"],
            "validation_base_rate": metrics["base_rate"],
            "validation_majority_accuracy": metrics["majority_accuracy"],
            "uses_market_context": self.market_context is not None,
            "is_calibrated": bool(self.model.is_calibrated),
            "calibration_method": self.model.calibration_method,
            "calibration_samples": int(len(X_calibrate)),
            "calibration_positives": int(np.sum(y_calibrate)) if len(X_calibrate) else 0,
            "expected_calibration_error": calibration_error,
            "mean_positive_probability": float(y_val_proba[:, 1].mean()),
            "feature_importance": feature_importance,
            "model_type": "hybrid_random_forest_xgboost",
            "decision_threshold": self.decision_threshold,
            "scale_pos_weight": self.scale_pos_weight,
            "swing_threshold": self.swing_threshold,
            "effective_swing_threshold": effective_threshold(self.swing_threshold),
            "lookforward_periods": self.lookforward_periods,
            "min_hold_periods": self.min_hold_periods,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "embargoed_samples": int(self._embargoed_rows),
        }
        self.is_trained = True
        return val_score

    @staticmethod
    def _expected_calibration_error(y_true, positive_proba, bins=10):
        """Mean gap between predicted probability and observed frequency, weighted by how
        many rows land in each probability bin.

        The number the calibration step exists to reduce, and the one that says whether a
        threshold can be reasoned about rather than searched for: if rows predicted 0.30
        come true 30% of the time, expected value is computable.
        """
        y_true = np.asarray(y_true, dtype=float)
        positive_proba = np.asarray(positive_proba, dtype=float)
        if len(y_true) == 0:
            return float("nan")
        edges = np.linspace(0.0, 1.0, bins + 1)
        total_error = 0.0
        for lower, upper in zip(edges[:-1], edges[1:]):
            in_bin = (positive_proba > lower) & (positive_proba <= upper)
            if not in_bin.any():
                continue
            total_error += in_bin.sum() * abs(y_true[in_bin].mean() - positive_proba[in_bin].mean())
        return float(total_error / len(y_true))

    @staticmethod
    def _validation_metrics(y_true, y_pred, positive_proba):
        """Threshold-free ranking quality (PR-AUC, ROC-AUC) plus the operating point
        the deployed decision_threshold actually lands on, and the accuracy an
        always-negative model would get on the same rows."""
        base_rate = float(np.mean(y_true)) if len(y_true) else 0.0
        has_both_classes = 0 < base_rate < 1
        return {
            "pr_auc": float(average_precision_score(y_true, positive_proba)) if has_both_classes else float("nan"),
            "roc_auc": float(roc_auc_score(y_true, positive_proba)) if has_both_classes else float("nan"),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "base_rate": base_rate,
            "majority_accuracy": 1.0 - base_rate,
        }

    def hyperparameters(self):
        return {
            "random_forest": self.random_forest_model.get_params(),
            "xgboost": self.xgboost_model.get_params(),
            "decision_threshold": self.decision_threshold,
            "swing_threshold": self.swing_threshold,
            "effective_swing_threshold": effective_threshold(self.swing_threshold),
            "lookforward_periods": self.lookforward_periods,
            "min_hold_periods": self.min_hold_periods,
            "scale_pos_weight": self.scale_pos_weight,
        }
