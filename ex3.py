import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Modules imported successfully")

datainit = load_breast_cancer()

df = pd.DataFrame(datainit.data, columns=datainit.feature_names)

X = df[['mean radius']]

df['target'] = datainit.target
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Prediction probabilities:")

print(model.score(X_test, y_test))

print(classification_report(y_test, pred))

plt.scatter(X['mean radius'], y, label='Actual Data')

X_val = pd.DataFrame(
    np.linspace(
        X['mean radius'].min(),
        X['mean radius'].max(),
        100
    ),
    columns=['mean radius']
)

y_val = model.predict_proba(X_val)[:, 1]

plt.plot(X_val, y_val, label='Logistic Curve')

plt.xlabel("Mean Radius")
plt.ylabel("Probability of Benign")
plt.title("Logistic Regression - Breast Cancer")
plt.legend()
plt.grid(True)
plt.show()
