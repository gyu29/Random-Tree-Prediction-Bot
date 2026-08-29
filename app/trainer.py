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

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

INTERNAL_VALIDATION_FRACTION = 0.15

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
    ):
        if XGBClassifier is None:
            raise ImportError("xgboost is required for training. Install it with: pip install xgboost")
        self.swing_threshold = swing_threshold
        self.lookforward_periods = lookforward_periods
        self.min_hold_periods = min_hold_periods
        self.decision_threshold = decision_threshold

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
            df_features = TechnicalIndicators.create_all_indicators(symbol_frame)
            df_labeled = create_swing_labels(
                df_features, self.swing_threshold, self.lookforward_periods, self.min_hold_periods
            )
            # Fill gaps against this symbol's own history, before the concat. Filling
            # afterwards lets one symbol supply another's values: a symbol with too
            # little history for a 200-day window has no sma_200/price_sma_200_ratio
            # column at all, and a frame-wide ffill/bfill hands it the neighbouring
            # symbol's numbers instead of leaving it missing. Category-wide calendar
            # cutoffs (scripts/build_factor_datasets.py) make short per-symbol training
            # frames routine, so this is a live case, not a theoretical one.
            fill_cols = [col for col in df_labeled.columns if col not in EXCLUDE_FROM_FEATURES]
            df_labeled[fill_cols] = df_labeled[fill_cols].ffill().bfill()
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
        """Per-symbol chronological split with an embargo, most recent
        INTERNAL_VALIDATION_FRACTION held out -- see module docstring for why this
        isn't a random shuffle, and why a bare chronological cut still isn't enough.

        The `lookforward_periods` rows immediately before each symbol's cut are
        dropped from both sides: their swing labels are computed from bars that fall
        on the validation side, so training on them lets the model see the outcome of
        the very rows it is about to be scored on.

        Masks are positional, not index-label based. df_clean's index is the date
        index and symbols share dates, so `mask.loc[one_symbol_slice.index] = False`
        also flips every other symbol's rows on those dates.
        """
        symbols = df_clean["symbol"].to_numpy()
        train_mask = np.zeros(len(df_clean), dtype=bool)
        validation_mask = np.zeros(len(df_clean), dtype=bool)
        embargoed = 0

        for symbol in np.unique(symbols):
            positions = np.flatnonzero(symbols == symbol)
            split_at = int(len(positions) * (1 - INTERNAL_VALIDATION_FRACTION))
            embargo_at = max(0, split_at - self.lookforward_periods)
            train_mask[positions[:embargo_at]] = True
            validation_mask[positions[split_at:]] = True
            embargoed += split_at - embargo_at

        self._embargoed_rows = embargoed
        return X[train_mask], X[validation_mask], y[train_mask], y[validation_mask]

    def train(self, data_directory):
        df = self.load_historical_data(data_directory)
        if len(df) < 500:
            raise ValueError("Insufficient data for training (minimum 500 samples required)")

        X, y, df_clean = self.prepare_training_data(df)
        print(f"Training dataset shape: {X.shape}")
        print(f"Swing opportunities: {sum(y)} out of {len(y)} samples ({sum(y) / len(y) * 100:.2f}%)")
        if sum(y) < 10:
            print("WARNING: very few positive samples -- consider a lower swing threshold or more data.")

        X_train, X_val, y_train, y_val = self._chronological_internal_split(df_clean, X, y)
        print(f"Internal split: {len(X_train)} train rows, {len(X_val)} validation rows "
              f"(most recent {INTERNAL_VALIDATION_FRACTION:.0%} per symbol), "
              f"{self._embargoed_rows} rows embargoed at the cut ({self.lookforward_periods} per symbol)")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.scale_pos_weight = compute_scale_pos_weight(y_train)
        self.xgboost_model.set_params(scale_pos_weight=self.scale_pos_weight)
        print(f"Computed scale_pos_weight={self.scale_pos_weight:.3f} from actual training label ratio")

        self.model.fit(X_train_scaled, y_train)
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        y_val_proba = self.model.predict_proba(X_val_scaled)

        train_score = accuracy_score(y_train, y_train_pred)
        val_score = accuracy_score(y_val, y_val_pred)
        metrics = self._validation_metrics(y_val, y_val_pred, y_val_proba[:, 1])

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
