from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class XGBStringClassifier(BaseEstimator, ClassifierMixin):
    """XGBClassifier wrapper that accepts and returns string class labels."""

    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1,
                 random_state=42, eval_metric='mlogloss', n_jobs=-1):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.eval_metric   = eval_metric
        self.n_jobs        = n_jobs

    def fit(self, X, y):
        import xgboost as xgb
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.model_ = xgb.XGBClassifier(
            n_estimators  = self.n_estimators,
            max_depth     = self.max_depth,
            learning_rate = self.learning_rate,
            random_state  = self.random_state,
            eval_metric   = self.eval_metric,
            n_jobs        = self.n_jobs,
        )
        self.model_.fit(X, y_enc)
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
