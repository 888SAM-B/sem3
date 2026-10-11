import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
iris = load_iris()
X = iris.data
y = iris.target
y = np.where(y == 0, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
accuracy = []
precision = []
recall = []
f1 = []
for k in kernels:
    svm = SVC(kernel=k, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f1.append(f)
    print("Kernel:", k)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f, 4))
x = np.arange(len(kernels))
width = 0.2
plt.bar(x - 0.3, accuracy, width, label='Accuracy')
plt.bar(x - 0.1, precision, width, label='Precision')
plt.bar(x + 0.1, recall, width, label='Recall')
plt.bar(x + 0.3, f1, width, label='F1 Score')
plt.xlabel("SVM Kernels")
plt.ylabel("Score")
plt.title("SVM Kernel Performance")
plt.xticks(x, kernels)
plt.legend()
plt.grid(axis='y')
plt.show()
