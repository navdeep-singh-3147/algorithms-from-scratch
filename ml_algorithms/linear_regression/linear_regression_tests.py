import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=4
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, colorizer="b", marker="o", s=30)
# plt.show()


from linear_regression import LiearRegression

regressor = LiearRegression(lr=0.05)
regressor.fit(X_train, y_train)
predicted_values = regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

mse_value = mse(y_true=y_test, y_pred=predicted_values)
print(mse_value)