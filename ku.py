import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_excel(r'D:\Desktop\tree,pca\credit.xlsx', engine='openpyxl')

# Print column names to verify the actual column names
print("Columns in the dataset:", df.columns)

# Rename columns for easier handling
column_rename_mapping = {
    '年龄段': 'AgeGroup',
    '有工作': 'HasJob',
    '有自己的房子': 'OwnsHouse',
    '信贷情况': 'CreditStatus',
    '类别(是否给贷款)': 'LoanApproval'
}
df.rename(columns=column_rename_mapping, inplace=True)

# Encoding categorical features manually using LabelEncoder
le = LabelEncoder()

# Apply label encoding for each categorical feature except 'ID'
for col in df.columns[1:]:  # Skip 'ID' column
    df[col] = le.fit_transform(df[col])

# Split dataset into features and target variable
X = df[['AgeGroup', 'HasJob', 'OwnsHouse', 'CreditStatus']]  # Features
y = df['LoanApproval']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Decision Tree Classifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Visualizing the Decision Tree
plt.figure(figsize=(14, 10))
plot_tree(dt,
          feature_names=X.columns,
          class_names=[str(c) for c in dt.classes_],
          filled=True,
          rounded=True,
          fontsize=10,
          proportion=True)

# Show the plot
plt.title('Decision Tree Visualization', fontsize=16)
plt.show()

# Make predictions on the test set
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')

# Optional: Show sample predictions
sample_predictions = pd.DataFrame({
    'Actual': y_test[:10].values,
    'Predicted': y_pred[:10]
})
print("\nSample predictions:")
print(sample_predictions)


