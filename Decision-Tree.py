import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def decision_tree_train(X, y):

    def build_tree(X, y):
        num_samples, num_features = X.shape
        if num_samples == 0:
            return None
        if len(set(y)) == 1:
            return DecisionNode(value=y[0])

        best_feature = None
        best_threshold = None
        best_gini = float('inf')

        for feature_index in range(num_features):
            thresholds = set(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                left_gini = gini_impurity(y[left_indices])
                right_gini = gini_impurity(y[right_indices])
                total_gini = (
                                     len(y[left_indices]) * left_gini +
                                     len(y[right_indices]) * right_gini
                             ) / num_samples

                if total_gini < best_gini:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_gini = total_gini

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_subtree = build_tree(X[left_indices], y[left_indices])
        right_subtree = build_tree(X[right_indices], y[right_indices])

        return DecisionNode(
            feature=best_feature, threshold=best_threshold,
            left=left_subtree, right=right_subtree
        )

    def gini_impurity(y):
        classes = set(y)
        impurity = 1
        for c in classes:
            p = len(y[y == c]) / len(y)
            impurity -= p ** 2
        return impurity

    return build_tree(X, y)


def plot_tree(node, feature_names, ax=None, x=0.5, y=1, dx=0.2, dy=0.1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

    if node.value is not None:
        ax.text(x, y, f"叶子:\nClass={node.value}", ha="center",
                bbox=dict(facecolor="lightgreen", edgecolor="black"), fontproperties=font)
        return

    feature = feature_names[node.feature]
    ax.text(x, y, f"{feature}\n<= {node.threshold:.2f}", ha="center",
            bbox=dict(facecolor="lightblue", edgecolor="black"), fontproperties=font)

    if node.left:
        ax.plot([x - dx, x], [y - dy, y], "k-")  # 左分支
        plot_tree(node.left, feature_names, ax=ax, x=x - dx, y=y - dy, dx=dx / 1.5, dy=dy)

    if node.right:
        ax.plot([x + dx, x], [y - dy, y], "k-")  # 右分支
        plot_tree(node.right, feature_names, ax=ax, x=x + dx, y=y - dy, dx=dx / 1.5, dy=dy)

if __name__ == "__main__":
    datasets = np.array([
        ['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否'],
        ['老年', '否', '否', '非常好', '否']
    ])
    datalabels = np.array(['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
    train_data = pd.DataFrame(datasets, columns=datalabels)

    X_train = pd.get_dummies(train_data.iloc[:, :-1]).values
    y_train = train_data.iloc[:, -1].map({'否': 0, '是': 1}).values

    feature_names = pd.get_dummies(train_data.iloc[:, :-1]).columns.tolist()

    tree = decision_tree_train(X_train, y_train)

    plot_tree(tree, feature_names=feature_names)
    plt.show()

