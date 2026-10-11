import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
df = pd.DataFrame(np.c_[iris.data, iris.target], columns=iris.feature_names + ['target'])
print(df.head())

plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=df['target'])
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset")
plt.show()

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris.data, iris.target)
knn_pred = knn.predict(iris.data)

print("KNN Accuracy:", knn.score(iris.data, iris.target))
print("KNN Confusion Matrix:\n", metrics.confusion_matrix(iris.target, knn_pred))

nb = GaussianNB()
nb.fit(iris.data, iris.target)
nb_pred = nb.predict(iris.data)

print("Naive Bayes Accuracy:", nb.score(iris.data, iris.target))
print("Naive Bayes Confusion Matrix:\n", metrics.confusion_matrix(iris.target, nb_pred))
