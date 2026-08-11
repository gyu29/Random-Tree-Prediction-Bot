"""Random Forest + XGBoost hybrid ensemble, and shared class-imbalance handling."""
import numpy as np


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


class HybridSwingEnsemble:
    """Average-probability ensemble over Random Forest and XGBoost."""

    def __init__(self, random_forest_model, xgboost_model, decision_threshold=0.70):
        self.random_forest_model = random_forest_model
        self.xgboost_model = xgboost_model
        self.decision_threshold = decision_threshold

    def fit(self, X, y):
        self.random_forest_model.fit(X, y)
        self.xgboost_model.fit(X, y)
        return self

    def predict_proba(self, X):
        rf_proba = self.random_forest_model.predict_proba(X)
        xgb_proba = self.xgboost_model.predict_proba(X)
        return (rf_proba + xgb_proba) / 2.0

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.decision_threshold).astype(int)

    @property
    def feature_importances_(self):
        rf_importance = np.asarray(self.random_forest_model.feature_importances_, dtype=float)
        xgb_importance = np.asarray(self.xgboost_model.feature_importances_, dtype=float)
        return (rf_importance + xgb_importance) / 2.0
