import numpy as np

# TODO: Implement the Decision Tree Classifier using NumPy
import numpy as np

class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

def gini_impurity(y):
    m = len(y)
    return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

def best_split(X, y):
    m, n = X.shape
    if m <= 1:
        return None, None

    # Unique classes and their mapping
    unique_classes = np.unique(y)
    class_map = {c: i for i, c in enumerate(unique_classes)}
    num_parent = [np.sum(y == c) for c in unique_classes]
    best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
    best_idx, best_thr = None, None

    for idx in range(n):
        thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
        num_left = [0] * len(unique_classes)
        num_right = num_parent.copy()
        for i in range(1, m):
            c = class_map[classes[i - 1]]
            num_left[c] += 1
            num_right[c] -= 1
            gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(len(unique_classes)))
            gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(len(unique_classes)))
            gini = (i * gini_left + (m - i) * gini_right) / m
            if thresholds[i] == thresholds[i - 1]:
                continue
            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = (thresholds[i] + thresholds[i - 1]) / 2

    return best_idx, best_thr

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        unique_classes = np.unique(y)
        num_samples_per_class = [np.sum(y == i) for i in unique_classes]
        predicted_class = unique_classes[np.argmax(num_samples_per_class)]
        node = DecisionTreeNode(
            gini=gini_impurity(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

# Now let's test the updated code
# Create a simple dataset
X_test = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [1, 0.5], [0.5, 1], [0.5, 0], [0, 0.5]])
y_test = np.array([0, 1, 1, 0, 1, 0, 1, 0])

# Create and train the classifier
clf_test = DecisionTreeClassifier(max_depth=3)
clf_test.fit(X_test, y_test)

# Test predictions
predictions_test = clf_test.predict(X_test)
print(predictions_test)
#output: [0, 1, 1, 0, 1, 0, 1, 0]
