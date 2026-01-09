from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "FCVC" in X.columns and "CH2O" in X.columns:
            X["Healthy_Habits"] = X["FCVC"] + X["CH2O"]

        if "FAF" in X.columns:
            X["Activity_Score"] = X["FAF"] * 2

        return X