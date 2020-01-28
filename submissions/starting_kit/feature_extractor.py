from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


numeric_cols = [c for c in suicide if suicide[c].dtype.kind in ('i', 'f')]
categorical_cols = [c for c in suicide if suicide[c].dtype.kind not in ('i', 'f')]
drop_cols = ['country-year']

numeric_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])

cat_encoder = make_column_transformer((OrdinalEncoder(), categorical_cols), remainder='passthrough')


def process_gdp_for_year(X):
    gdp = X[' gdp_for_year ($)'].str[:-12]
    return pd.to_numeric(gdp, errors='coerce').values[:, np.newaxis]

gdp_year_transformer = FunctionTransformer(process_gdp_for_year, validate=False)
   

ct = ColumnTransformer(
    transformers=[('numeric', numeric_transformer, numeric_cols),
                  ('categoric', cat_encoder, categorical_cols),
                  ('gdpforyear', gdp_year_transformer,['gdp_for_year ($)']),
                  ('drop cols', 'drop', drop_cols),
                 ])
