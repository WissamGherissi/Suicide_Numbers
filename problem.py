import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error

problem_title = 'Prediction of Suicide Numbers'
_target_column_name = 'suicides_no'

Predictions = rw.prediction_types.make_regression(
    label_names=['suicide_no'])

workflow = rw.workflows.FeatureExtractorRegressor()

class Suicide_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='suicide error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        loss = np.mean(np.abs(y_true.reshape(len(y_true),1)-y_pred.reshape(len(y_pred),1))*(y_true**2)/np.max(y_true**2))
        return loss


score_types = [
    Suicide_error(name='suicide error', precision=2),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.1)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[::30], y_array[::30].reshape(len(y_array[::30]),1)
    else:
        return X_df, y_array.reshape(len(y_array),1)
    return X_df, y_array.reshape(len(y_array),1)


def get_train_data(path='.'):
    f_name = 'suicide_train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'suicide_test.csv'
    return _read_data(path, f_name)
