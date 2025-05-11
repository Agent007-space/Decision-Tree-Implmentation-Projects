# Decision-Tree-Implmentation-Projects
import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # index of feature to split on
        self.threshold = threshold  # threshold value for the split
        self.left = left  # left child
        self.right = right  # right child
        self.value = value  # class prediction (for leaf nodes)

class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if depth >= self.max_depth or num_labels == 1 or num_samples == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        best_feature, best_thresh = self._best_split(X, y, num_features)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # Split
        left_idx = X[:, best_feature] <= best_thresh
        right_idx = X[:, best_feature] > best_thresh

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return DecisionTreeNode(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, num_features):
        best_gini = 1.0
        split_idx, split_thresh = None, None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for thresh in thresholds:
                left_idx = X[:, feature_index] <= thresh
                right_idx = X[:, feature_index] > thresh

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gini = self._gini_index(y[left_idx], y[right_idx])

                if gini < best_gini:
                    best_gini = gini
                    split_idx = feature_index
                    split_thresh = thresh

        return split_idx, split_thresh

    def _gini_index(self, left_labels, right_labels):
        def gini(labels):
            classes = np.unique(labels)
            impurity = 1.0
            for cls in classes:
                p = np.sum(labels == cls) / len(labels)
                impurity -= p ** 2
            return impurity

        total = len(left_labels) + len(right_labels)
        return (len(left_labels) / total) * gini(left_labels) + \
               (len(right_labels) / total) * gini(right_labels)

    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
