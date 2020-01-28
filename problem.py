import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error

problem_title = 'Prediction of Suicide Numbers'
_target_column_name = 'suicides_no'

Predictions = rw.prediction_types.make_regression(
    label_names=['suicide_no'])

workflow = rw.workflows.FeatureExtractorRegressor()

# define the score (specific score for the FAN problem)
class Suicide_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='suicide error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            loss = mean_squared_error(y_true,y_pred)
        return loss


score_types = [
    Suicide_error(name='suicide error', precision=2),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.20)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, '/data', f_name), low_memory=False)
    #data = data.sample(frac=0.1, replace=False, random_state=1)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[::30], y_array[::30]
    else:
        return X_df, y_array
    return X_df, y_array


def get_train_data(path='.'):
    f_name = '/Suicide_TRAIN.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = '/Suicide_TEST.csv'
    return _read_data(path, f_name)
