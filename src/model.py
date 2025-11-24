import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os

class FraudDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, scale_pos_weight=1.0, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y, **kwargs):
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_shap_values(self, X):
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath):
        model = FraudDetector()
        model.model = joblib.load(filepath)
        return model
