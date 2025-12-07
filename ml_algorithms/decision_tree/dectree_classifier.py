import numpy as np

def gini_impurity(y):
    """
    Calculate Gini impurity for a set of labels.
    """
    if len(y)==0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities**2)
    return gini

def information_gain(y, y_left, y_right):
    """
    Calculate information gain from a split.
    y: parent labels
    y_left, y_right: labels in child nodes
    """
    parent_gini = gini_impurity(y)
    n= len(y)
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left==0 or n_right==0:
        return 0
    weighted_gini = (n_left / n) * gini_impurity(y_left) + (n_right / n) * gini_impurity(y_right)
    return parent_gini - weighted_gini


# Select best feature and threshold
def best_split(X, y):
    """
    Find the best feature and threshold for splitting.
    X: featues (n_samples, n_features)
    y: labels
    Return: best_features_index, best_threshold, best_gain
    """
    n_samples, n_features = X.shape
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        if len(thresholds) < 2:
            continue
        for i in range(1, len(thresholds)):
            threshold = (thresholds[i-1] + thresholds[i]) / 2
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            y_left = y[left_mask]
            y_right = y[right_mask]
            gain = information_gain(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold, best_gain


# Node structure
class DecisionNode:
    """
    Represent a node in the decision tree
    """
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature_index=feature_index # Index of the feature to split on
        self.threshold=threshold # Threshold for split
        self.left=left # Left child node
        self.right=right # Right child node
        self.value=value # Class label for leaf node
        
        
# Recursive tree building
def build_tree(
    X, y,
    max_depth=10,
    min_samples_split=2,
    depth=0   
):
    n_samples = len(y)
    if n_samples < min_samples_split or depth >= max_depth or len(np.unique(y))==1:
        # Leaf node: majority class
        leaf_value = np.bincount(y).argmax()
        return DecisionNode(value=leaf_value)
    
    feature, threshold, gain = best_split(X, y)
    if gain == 0:
        # No gain: make leaf
        leaf_value = np.bincount(y).argmax()
        return DecisionNode(value=leaf_value)
    
    # Split data
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    
    # Recurse
    left_child = build_tree(X_left, y_left, max_depth, min_samples_split, depth + 1)
    right_child = build_tree(X_right, y_right, max_depth, min_samples_split, depth + 1)
    
    return DecisionNode(feature_index=feature, threshold=threshold, left=left_child, right=right_child)
    
# Prediction helper function
def predict(node, x):
    """
    Predict function for single sample.
    """
    if node.value is not None:
        return node.value
    if x[node.feature_index] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)
    

def predict_batch(root, X):
    """
    Predict for multiple samples.
    """
    return np.array([predict(root, x) for x in X])

class DecisionTreeClassifier:
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        self.root = build_tree(X, y, self.max_depth, self.min_samples_split)
        
    def predict(self, X):
        if self.root is None:
            raise ValueError("Tree not fitted yet!")
        return predict_batch(self.root, X)
    
    
if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    iris = load_iris()
    # X, y = iris.data, iris.target # Binary keep only two classes for simplicity
    
    X, y = make_classification(n_samples=2000, n_features=30, random_state=4)
    X, y = X[y<2], y[y<2]
    print(f"Shape: ({X.shape}, {len(y)})")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    clf = DecisionTreeClassifier(max_depth=1, min_samples_split=10)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    print(f"Accuracy score: {accuracy:.2f}")
    