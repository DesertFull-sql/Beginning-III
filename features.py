from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from config import NUMERIC_FEATURES


def build_numeric_pipeline(
        impute_strategy: str = 'median'
) -> Pipeline:
    return Pipeline(
        [
            ('fill_nan', SimpleImputer(strategy=impute_strategy)),
            ('scaler', StandardScaler())
        ]
    )


def build_categorical_pipeline(
        impute_strategy: str = 'most_frequent'
) -> Pipeline:
    return Pipeline(
        [
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))    
        ]
    )

def build_preprocessor(
        num_columns,
        cat_columns
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ('num', build_numeric_pipeline(), num_columns),
            ('cat', build_categorical_pipeline(), cat_columns)
        ],
        remainder='passthrough'
    )
    
