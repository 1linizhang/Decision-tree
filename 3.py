import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset from the provided file path
# Ensure openpyxl is installed: pip install openpyxl
df = pd.read_excel(r'D:\Desktop\tree,pca\credit.xlsx', engine='openpyxl')

# Print column names to verify the actual column names
print("Columns in the dataset:", df.columns)

# Rename columns from Chinese to English for easier handling
column_rename_mapping = {
    '年龄段': 'AgeGroup',
    '有工作': 'HasJob',
    '有自己的房子': 'OwnsHouse',
    '信贷情况': 'CreditStatus',
    '类别(是否给贷款)': 'LoanApproval'
}
df.rename(columns=column_rename_mapping, inplace=True)


# Encoding the categorical features manually
def encode_feature(col):
    mapping = {val: idx for idx, val in enumerate(col.unique())}
    return col.map(mapping)


# Encoding all features
for col in df.columns[1:]:  # Skip 'ID' column
    df[col] = encode_feature(df[col])

# Adjusting the column names to match the actual dataset
X = df[['AgeGroup', 'HasJob', 'OwnsHouse', 'CreditStatus']]
Y = df['LoanApproval']

# Splitting the dataset into training and testing data
split = int(0.7 * len(X))
X_train, X_test = X[:split].reset_index(drop=True), X[split:].reset_index(drop=True)
Y_train, Y_test = Y[:split].reset_index(drop=True), Y[split:].reset_index(drop=True)


# Function to calculate entropy
def entropy(data, label):
    label_counts = data[label].value_counts()
    total = len(data)
    ent = 0
    for count in label_counts:
        p = count / total
        if p > 0:
            ent += -p * np.log2(p)
    return ent


# Function to split dataset based on feature and value
def split_data_set(data, feature, value):
    left_data = data[data[feature] <= value]
    right_data = data[data[feature] > value]
    return left_data, right_data


# Function to choose the best feature and split value based on information gain
def choose_feature(data, label):
    base_entropy = entropy(data, label)
    best_info_gain = -float('inf')
    best_feature = None
    best_split_value = None

    # Iterate over all features to find the best one
    for col in data.columns:
        if col == label:
            continue  # Skip the label column

        # Sorting values of the feature
        col_unique = np.sort(data[col].unique())
        for split_value in col_unique:
            # Split the dataset based on the current feature and value
            left_data, right_data = split_data_set(data, col, split_value)

            # Calculate weighted entropy after the split
            p_left = len(left_data) / len(data)
            p_right = len(right_data) / len(data)
            new_entropy = p_left * entropy(left_data, label) + p_right * entropy(right_data, label)

            # Calculate information gain
            info_gain = base_entropy - new_entropy
            # Track the best feature and best split value
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = col
                best_split_value = split_value

    return best_feature, best_split_value


# Recursive function to build a simple decision tree
def build_tree(data, label, max_depth, depth=0):
    # Stopping condition for recursion
    if depth >= max_depth or len(data[label].unique()) == 1 or len(data) == 0:
        return data[label].mode()[0] if len(data) > 0 else None

    # Choose the best feature and the corresponding split value
    best_feature, best_value = choose_feature(data, label)

    print(f"Depth {depth}: Best Feature: {best_feature}, Split Value: {best_value}")  # Debugging line

    # Split the dataset into two parts based on the best feature and best value
    left_data, right_data = split_data_set(data, best_feature, best_value)

    # Recursively build the decision tree
    tree = {}
    tree[best_feature] = {}
    tree[best_feature]['<= {}'.format(best_value)] = build_tree(left_data, label, max_depth, depth + 1)
    tree[best_feature]['> {}'.format(best_value)] = build_tree(right_data, label, max_depth, depth + 1)

    return tree


# Build the decision tree using the training dataset
decision_tree = build_tree(pd.concat([X_train, Y_train], axis=1), 'LoanApproval', max_depth=3)
print("Decision Tree:")
print(decision_tree)


# Function to predict using the decision tree
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    for key in tree[feature].keys():
        value = float(key.split(' ')[1])
        if ('<=' in key and sample[feature] <= value) or ('>' in key and sample[feature] > value):
            return predict(tree[feature][key], sample)


# Predict on test dataset and calculate accuracy
y_pred = [
    predict(decision_tree, X_test.iloc[i]) if predict(decision_tree, X_test.iloc[i]) is not None else Y_train.mode()[0]
    for i in range(len(X_test))]

accuracy = np.sum(np.array(y_pred) == np.array(Y_test)) / len(Y_test)
print("Accuracy:", accuracy)


# Plotting the Decision Tree with Categorical Values
def plot_tree_manual(tree, depth=0, pos=(0, 0), parent=None, ax=None, x_offset=0.2, y_offset=0.2):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

    if not isinstance(tree, dict):
        ax.text(pos[0], pos[1], f"Leaf\nClass = {tree}", ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='lightblue', edgecolor='black'))
        if parent is not None:
            ax.plot([parent[0], pos[0]], [parent[1], pos[1]], color='black')
        return

    # Get the current node's feature and split condition
    feature = list(tree.keys())[0]
    children = tree[feature]

    for key, subtree in children.items():
        # Extracting the condition (<= or >) and the value
        split_condition = key
        value = float(split_condition.split(' ')[1])

        # Plotting the current node with the condition
        ax.text(pos[0], pos[1], f"{feature}\n{split_condition} ({value})", ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='orange', edgecolor='black'))

        # Plotting the lines connecting nodes
        if parent is not None:
            ax.plot([parent[0], pos[0]], [parent[1], pos[1]], color='black')

        # Position for child nodes
        left_pos = (pos[0] - x_offset, pos[1] - y_offset)
        right_pos = (pos[0] + x_offset, pos[1] - y_offset)

        # Recursively plot left and right branches
        if '<=' in split_condition:
            plot_tree_manual(subtree, depth + 1, pos=left_pos, parent=pos, ax=ax, x_offset=x_offset / 1.5,
                             y_offset=y_offset)
        elif '>' in split_condition:
            plot_tree_manual(subtree, depth + 1, pos=right_pos, parent=pos, ax=ax, x_offset=x_offset / 1.5,
                             y_offset=y_offset)

    if depth == 0:
        plt.show()


# Plot the manually built decision tree
plot_tree_manual(decision_tree)


