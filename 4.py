import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 替换为适合您系统的字体路径

class Node(object):
    def __init__(self, x=None, label=None, y=None, data=None):
        self.label = label   # label: 子节点分类依据的特征
        self.x = x           # x: 特征值
        self.child = []      # child: 子节点
        self.y = y           # y: 类标记（叶节点才有）
        self.data = data     # data: 包含数据（叶节点才有）

    def append(self, node):  # 添加子节点
        self.child.append(node)

    def predict(self, features):  # 预测数据所属类
        if self.y is not None:
            return self.y
        for c in self.child:
            if c.x == features[self.label]:
                return c.predict(features)


class DTreeID3(object):
    def __init__(self, epsilon=0, alpha=0):
        # 信息增益阈值
        self.epsilon = epsilon
        self.alpha = alpha
        self.tree = Node()

    def prob(self, datasets):
        datalen = len(datasets)
        labelx = set(datasets)
        p = {l: 0 for l in labelx}
        for d in datasets:
            p[d] += 1
        for i in p.items():
            p[i[0]] /= datalen
        return p

    def calc_ent(self, datasets):
        p = self.prob(datasets)
        value = list(p.values())
        return -np.sum(np.multiply(value, np.log2(value)))

    def cond_ent(self, datasets, col):
        labelx = set(datasets.iloc[col])
        p = {x: [] for x in labelx}
        for i, d in enumerate(datasets.iloc[-1]):
            p[datasets.iloc[col][i]].append(d)
        return sum([self.prob(datasets.iloc[col])[k] * self.calc_ent(p[k]) for k in p.keys()])

    def info_gain_train(self, datasets, datalabels):
        datasets = datasets.T
        ent = self.calc_ent(datasets.iloc[-1])
        gainmax = {}
        for i in range(len(datasets) - 1):
            cond = self.cond_ent(datasets, i)
            gainmax[ent - cond] = i
        m = max(gainmax.keys())
        return gainmax[m], m

    def train(self, datasets, node):
        labely = datasets.columns[-1]
        if len(datasets[labely].value_counts()) == 1:
            node.data = datasets[labely]
            node.y = datasets[labely][0]
            return
        if len(datasets.columns[:-1]) == 0:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return
        gainmaxi, gainmax = self.info_gain_train(datasets, datasets.columns)
        if gainmax <= self.epsilon:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return
        vc = datasets[datasets.columns[gainmaxi]].value_counts()
        for Di in vc.index:
            node.label = gainmaxi
            child = Node(Di)
            node.append(child)
            new_datasets = pd.DataFrame([list(i) for i in datasets.values if i[gainmaxi] == Di], columns=datasets.columns)
            self.train(new_datasets, child)

    def fit(self, datasets):
        self.train(datasets, self.tree)


def plot_tree(node, feature_names, x=0.5, y=1, dx=0.1, dy=0.1, ax=None, depth=0):
    """
    Recursively plots the decision tree using matplotlib, including specific values for each classification.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

    if node.y is not None:  # Leaf node
        counts = Counter(node.data)  # Count specific class values in this leaf
        count_text = "\n".join([f"{k}: {v}" for k, v in counts.items()])
        ax.text(x, y, f"叶子: {node.y}\n{count_text}", ha="center",
                fontproperties=font, bbox=dict(facecolor="lightgreen", edgecolor="black"))
    else:  # Internal node
        feature_label = feature_names[node.label]
        ax.text(x, y, f"特征: {feature_label}", ha="center", fontproperties=font,
                bbox=dict(facecolor="lightblue", edgecolor="black"))

        # Plot child nodes
        num_children = len(node.child)
        child_x_positions = np.linspace(x - dx * (num_children - 1) / 2, x + dx * (num_children - 1) / 2, num_children)

        for i, child in enumerate(node.child):
            child_x = child_x_positions[i]
            ax.plot([x, child_x], [y - 0.02, y - dy + 0.02], "k-")  # Connect parent to child
            ax.text((x + child_x) / 2, y - dy / 2, f"值: {child.x}", ha="center",
                    fontproperties=font, fontsize=10)  # Add condition values
            plot_tree(child, feature_names, x=child_x, y=y - dy, dx=dx / 1.5, dy=dy, ax=ax, depth=depth + 1)


if __name__ == "__main__":
    # 从 Excel 文件中加载数据
    file_path = r'D:\Desktop\tree,pca\credit.xlsx'  # 替换为实际文件路径
    df = pd.read_excel(file_path)

    # 提取特征和目标列
    datasets = df.values  # 转换为 NumPy 数组
    datalabels = df.columns  # 提取列名
    train_data = pd.DataFrame(datasets, columns=datalabels)

    dt = DTreeID3(epsilon=0)
    dt.fit(train_data)

    # 显示决策树
    feature_names = datalabels[:-1]  # Features for tree labels
    plot_tree(dt.tree, feature_names)
    plt.show()
