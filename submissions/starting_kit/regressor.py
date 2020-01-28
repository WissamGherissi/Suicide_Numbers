from sklearn import linear_model
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.rgr = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=.5))  #Avec scaling

    def fit(self, X, y):
        self.rgr.fit(X, y.reshape((len(y), 1)))

    def predict(self, X):
        y_pred = self.rgr.predict(X)
        return y_pred.reshape((len(y_pred), 1))
