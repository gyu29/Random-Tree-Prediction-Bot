"""Random Forest + XGBoost hybrid ensemble, class-imbalance handling, and calibration.

Averaging two probability estimates only means something if both are on the same scale,
and neither of these is by default. A random forest's positive-class score is the share
of trees voting yes, which is systematically pulled toward the middle; gradient boosting
under `scale_pos_weight` is pushed the other way, reporting confident positives far more
often than they occur. Averaging the two produced a number between 0 and 1 that was not
a probability of anything, which is why the eight deployed decision thresholds had to be
searched for individually and landed anywhere between 0.10 and 0.70 -- that spread was a
symptom, not eight considered choices about risk.

So each member is calibrated against held-out data before the average is taken, and the
calibration slice is purged from the fitting slice the same way every other boundary in
this project is (see app/trainer.py). After that, 0.30 means roughly a 30% chance, the
same way in every category, and the entry threshold becomes something computable from
expected value rather than something swept for on a grid --
scripts/expected_value_thresholds.py does that arithmetic.

`fit()` (uncalibrated) is kept because model_registry loads pickled ensembles trained
before this existed, and because the calibrated path needs enough positives in the
calibration slice to estimate anything.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Isotonic regression is non-parametric and fits any monotonic distortion, but it needs
# enough positives to do that without memorizing them. Below this count the far more
# constrained sigmoid (Platt) fit is the safer estimator -- it has two parameters, so it
# cannot chase noise the way a step function can.
MIN_POSITIVES_FOR_ISOTONIC = 100

# Below this, calibration is skipped entirely rather than fitted on almost nothing. A
# mapping estimated from a handful of positives is not merely imprecise, it is
# confidently wrong, and downstream code treats a calibrated probability as a real one --
# scripts/expected_value_thresholds.py computes an entry threshold from it. Declining to
# calibrate leaves the raw scores visibly uncalibrated, which is the honest state.
MIN_POSITIVES_FOR_CALIBRATION = 20


def compute_scale_pos_weight(y):
    """XGBoost's documented imbalance correction: negative/positive count ratio,
    computed from the actual training labels rather than a fixed guess. RandomForest
    gets the same "adapt to the real ratio" treatment via class_weight='balanced_subsample'
    (already data-driven per bootstrap sample) -- this is XGBoost's equivalent so neither
    ensemble member is using a number disconnected from the data.
    """
    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


def choose_calibration_method(y_calibrate):
    """isotonic when the calibration slice can support it, sigmoid otherwise."""
    positives = int(np.sum(np.asarray(y_calibrate) == 1))
    return "isotonic" if positives >= MIN_POSITIVES_FOR_ISOTONIC else "sigmoid"


class HybridSwingEnsemble:
    """Average-probability ensemble over Random Forest and XGBoost.

    When `fit_calibrated` was used, the average is taken over each member's *calibrated*
    probability; `is_calibrated` says which. predict_proba is the only thing callers need
    to care about either way.
    """

    def __init__(self, random_forest_model, xgboost_model, decision_threshold=0.70):
        self.random_forest_model = random_forest_model
        self.xgboost_model = xgboost_model
        self.decision_threshold = decision_threshold
        self.calibrated_random_forest = None
        self.calibrated_xgboost = None
        self.calibration_method = None

    @property
    def is_calibrated(self):
        return self.calibrated_random_forest is not None and self.calibrated_xgboost is not None

    def fit(self, X, y):
        """Uncalibrated fit. Kept for backward compatibility; prefer fit_calibrated."""
        self.random_forest_model.fit(X, y)
        self.xgboost_model.fit(X, y)
        self.calibrated_random_forest = None
        self.calibrated_xgboost = None
        self.calibration_method = None
        return self

    def fit_calibrated(self, X_fit, y_fit, X_calibrate, y_calibrate):
        """Fits both members on (X_fit, y_fit), then maps each one's raw scores onto real
        probabilities using (X_calibrate, y_calibrate).

        cv="prefit" is deliberate: the alternative, letting CalibratedClassifierCV split
        internally, uses stratified shuffled folds, which on overlapping rolling-window
        features would leak adjacent windows across the calibration boundary -- the same
        defect the chronological split in app/trainer.py exists to avoid. The caller is
        responsible for having purged X_calibrate from X_fit.

        Falls back to the uncalibrated fit when the calibration slice holds fewer than
        MIN_POSITIVES_FOR_CALIBRATION positives -- see that constant for why a thin
        calibration is worse than none.
        """
        self.random_forest_model.fit(X_fit, y_fit)
        self.xgboost_model.fit(X_fit, y_fit)

        positives = int(np.sum(np.asarray(y_calibrate) == 1))
        if positives < MIN_POSITIVES_FOR_CALIBRATION or len(np.unique(y_calibrate)) < 2:
            self.calibrated_random_forest = None
            self.calibrated_xgboost = None
            self.calibration_method = None
            return self

        method = choose_calibration_method(y_calibrate)
        self.calibration_method = method
        self.calibrated_random_forest = CalibratedClassifierCV(
            self.random_forest_model, method=method, cv="prefit"
        ).fit(X_calibrate, y_calibrate)
        self.calibrated_xgboost = CalibratedClassifierCV(
            self.xgboost_model, method=method, cv="prefit"
        ).fit(X_calibrate, y_calibrate)
        return self

    def predict_proba(self, X):
        if self.is_calibrated:
            rf_proba = self.calibrated_random_forest.predict_proba(X)
            xgb_proba = self.calibrated_xgboost.predict_proba(X)
        else:
            rf_proba = self.random_forest_model.predict_proba(X)
            xgb_proba = self.xgboost_model.predict_proba(X)
        return (rf_proba + xgb_proba) / 2.0

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.decision_threshold).astype(int)

    @property
    def feature_importances_(self):
        """Always read off the base estimators -- calibration wraps a model's outputs and
        has no importances of its own."""
        rf_importance = np.asarray(self.random_forest_model.feature_importances_, dtype=float)
        xgb_importance = np.asarray(self.xgboost_model.feature_importances_, dtype=float)
        return (rf_importance + xgb_importance) / 2.0
