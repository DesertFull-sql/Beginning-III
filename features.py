from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from config import NUMERIC_FEATURES


def build_numeric_transformer(
        columns,
        impute_strategy: str = 'median'
) -> ColumnTransformer:
    numeric_imputer = ColumnTransformer(
        transformers=[
            ('fill_nan', SimpleImputer(strategy=impute_strategy), columns)
        ],
        remainder='passthrough'
    )
    return numeric_imputer 


def build_categorical_pipeline(
        impute_strategy: str = 'most_frequent'
) -> Pipeline:
    categorical_imputer = Pipeline(
        [
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))    
        ]
    )
    return categorical_imputer
    
