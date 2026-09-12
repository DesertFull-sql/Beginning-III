from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from features import build_preprocessor


def build_log_reg(
        impute_max_iter: int = 1000
) -> Pipeline:
    return Pipeline(
        [
            ('preprocessor', build_preprocessor()),
            ('model',
                LogisticRegression(
                    max_iter=impute_max_iter,
                    class_weight='balanced'
                )
             )
        ]
    )

