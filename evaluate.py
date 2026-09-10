import joblib
from config import TARGET, TEST_SIZE, RANDOM_STATE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



model = joblib.load('model.pkl')

d = load_breast_cancer(as_frame=True)

df = d.frame


X = df.drop(columns=TARGET)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

y_proba = model.predict(X_test)
print(classification_report(y_test, y_proba))