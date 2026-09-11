from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate, train_test_split
from config import TARGET, TEST_SIZE, RANDOM_STATE
from models import logreg
import pandas as pd
import joblib


def train_model():

    d = load_breast_cancer(as_frame=True)
    df = d.frame

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    scoping = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_results = cross_validate(
        logreg,
        X_train,
        y_train,
        cv=5,
        scoring=scoping
    )

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cv_results[f'test_{metric}']
        print(f"{metric:<10}: mean={scores.mean():.3f}  std={scores.std():.3f}")

    logreg.fit(X_train, y_train)

    joblib.dump(logreg, 'model.pkl')