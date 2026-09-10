from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd

d = load_breast_cancer(as_frame=True)

df = d.frame


X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# pipeline = Pipeline([
#     ('impute', ColumnTransformer([
#         ('fillna', SimpleImputer(strategy='median'), NUMERIC_FEATURES)
#     ], remainder='passthrough')),
#     ('scaler', StandardScaler())
# ])

# preprocessor = ColumnTransformer([
#     ('num', pipeline, pipeline)
# ])

log_reg = Pipeline([
    ('fillna', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    # ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        class_weight='balanced'
    ))
])

scoping = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    log_reg,
    X_train,
    y_train,
    cv=5,
    scoring=scoping
)

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    scores = cv_results[f'test_{metric}']
    print(f"{metric:<10}: mean={scores.mean():.3f}  std={scores.std():.3f}")