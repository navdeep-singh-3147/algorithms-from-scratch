import numpy as np

def mse(y):
    if len(y)==0:
        return 0
    mean_y = np.mean(y)
    return np.mean((y - mean_y)**2)


def variance_reduction(y, y_left, y_right):
    parent_mse = mse(y)
    n, n_left, n_right = len(y), len(y_left), len(y_right)
    if n_left == 0 or n_right == 0:
        return 0
    weighted_mse = (n_left / n) * mse(y_left) + (n_right / n) * mse(y_right)
    return parent_mse - weighted_mse

def best_split_regressor(X, y):
    n_samples, n_feautres = X.shape
    best_reduction = -1
    best_feature = None
    best_threshold = None
    
    for feature in range(n_feautres):
        thresholds = np.unique(X[:, feature])
        if len(thresholds) < 2:
            continue
        for i in range(1, len(thresholds)):
            threshold = (thresholds[i-1] + thresholds[i]) / 2
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            y_left, y_right = y[left_mask], y[right_mask]
            reduction = variance_reduction(y, y_left, y_right)
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold, best_reduction


# Node structure for regression
class RegressionNode:
    
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        variance=None
    ):
        self.feature_index=feature_index # Feature to split on
        self.threshold=threshold # Split threshold
        self.left=left # Left child node
        self.right=right # Right child node
        self.value=value # Predicted value (mean for leaf)
        self.variance=variance # Variance of values in this node

# Recursive tree building        
def build_regression_tree(
    X, y,
    max_depth=10,
    min_samples_split=2,
    min_variance_reduction=0.0,
    depth=0
):
    n_samples = len(y)
    node_variance = mse(y)
    
    # Stopping criteria: make leaf node
    if (n_samples < min_samples_split
        or depth >= max_depth
        or node_variance < 1e-7): # Already homogeneous
        leaf_value = np.mean(y)
        return RegressionNode(value=leaf_value, variance=node_variance)
    
    # Find best split
    feature, threshold, reduction = best_split_regressor(X, y)
    
    if reduction < min_variance_reduction:
        # Not enough improvement
        leaf_value = np.mean(y)
        return RegressionNode(value=leaf_value, variance=node_variance)
    
    # Partition data
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    
    # Recursively build children
    left_child = build_regression_tree(
        X_left, y_left,
        max_depth,
        min_samples_split,
        min_variance_reduction,
        depth+1
    )
    
    right_child = build_regression_tree(
        X_right, y_right,
        max_depth,
        min_samples_split,
        min_variance_reduction,
        depth+1
    )
    
    return RegressionNode(
        feature_index=feature,
        threshold=threshold,
        left=left_child,
        right=right_child,
        variance=node_variance
    )
    
    
def predict_single(node, x):
    """
    Predict continuous value for a single sample.
    """
    if node.value is not None:
        return node.value
    
    if x[node.feature_index] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)
    
def predict_batch(root, X):
    """
    Predict for multiple samples
    """
    return np.array([predict_single(root, x) for x in X])


# prediction variance (Confidence estimation)
def predict_with_variance(node, x):
    if node.value is not None:
        return node.value, node.variance
    
    if x[node.feature_index] <= node.threshold:
        return predict_with_variance(node.left, x)
    else:
        return predict_with_variance(node.right, x)
    
def predict_batch_with_variance(root, X):
    """
    Predict values and variances for multiple samples
    """
    results = [predict_with_variance(root, x) for x in X]
    predictions = np.array([r[0] for r in results])
    variances = np.array([r[1] for r in results])
    return predictions, variances
    
class DecisionTreeRegressor:
    
    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        min_variance_reduction=0.0
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_variance_reduction = min_variance_reduction
        self.root = None
        
    def fit(self, X, y):
        self.root = build_regression_tree(
            X, y,
            self.max_depth,
            self.min_samples_split,
            self.min_variance_reduction
        )
        return self
    
    def predict(self, X):
        if self.root is None:
            raise ValueError("model not fitted yet.")
        return predict_batch(self.root, X)
    
    def predict_with_variance(self, X):
        if self.root is None:
            raise ValueError("model not fitted yet.")
        return predict_batch_with_variance(self.root, X)
    
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    # Generate synthetic regression data
    X, y = make_regression(n_samples=2000, n_features=1, noise=10, random_state=42)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train regressor
    regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_variance_reduction=0.1)
    regressor.fit(X_train, y_train)
    
    # Predict
    y_pred = regressor.predict(X_test)

    # Evaluate
    mse_score = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse_score:.2f}")
    print(f"R² Score: {r2:.3f}")

    # Predict with variance
    y_pred_var, variances = regressor.predict_with_variance(X_test)
    print(f"\nPredictions with variance for first 5 samples:")
    for i in range(5):
        print(f"  Predicted: {y_pred_var[i]:.2f}, Variance: {variances[i]:.2f}")