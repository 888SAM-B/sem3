import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Type of X:", type(X))
print("Shape of X:", X.shape)
print("Type of Cluster Labels:", type(labels))
print("Shape of Cluster Labels:", labels.shape)
print("Type of Cluster Centers:", type(centers))
print("Shape of Cluster Centers:", centers.shape)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0],  centers[:, 1], c='red', marker='x', s=200)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("K-Means Clustering")
plt.show()
