from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from config import NUMERIC_FEATURES

numeric_transformer = ColumnTransformer(
    transformers=[
        ('fill_nan', SimpleImputer(strategy='median'), NUMERIC_FEATURES)
    ],
    remainder='passthrough'
)

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, NUMERIC_FEATURES),
    # ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
])