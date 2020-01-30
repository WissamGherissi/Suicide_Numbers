from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import category_encoders as ce
import pandas as pd
import numpy as np


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_new = X_df.copy()
        numeric_cols = [c for c in X_new if X_new[c].dtype.kind in ('i', 'f') ]
        drop_cols = ['country-year','continent']
        ct = ColumnTransformer(transformers=[('numeric', numeric_transformer, numeric_cols),
          ('binary', OrdinalEncoder(), ['sex','age','generation']),                                 
          ('drop cols', 'drop', drop_cols),
         ])

        XX = ct.fit_transform(X_new)
        return XX
        
numeric_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
