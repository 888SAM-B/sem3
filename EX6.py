import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)

parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5]
}

grid = GridSearchCV(dt, parameters, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

model = grid.best_estimator_
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.title("Decision Tree Classification")
plt.show()
