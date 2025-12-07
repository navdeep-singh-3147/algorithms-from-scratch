import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_pred==y_true) / len(y_pred)
    return accuracy

from logistic_regression import LogisticRegression

regressor = LogisticRegression(lr=0.05)
regressor.fit(X_train, y_train)
predicted_classes = regressor.predict(X_test)

print(f"LR classification accuracy: {accuracy(y_true=y_test, y_pred=predicted_classes)}")