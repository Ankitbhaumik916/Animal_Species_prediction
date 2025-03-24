import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Zoo dataset
df = pd.read_csv("zoo.csv")  # Ensure you have the dataset in the same directory

# Drop the 'animal_name' column since it's not a feature
df = df.drop(columns=["animal_name"])

# Extract features and labels
X = df.drop(columns=["class_type"])
y = df["class_type"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train ID3 Decision Tree (using entropy as the criterion)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)  # Limiting depth to prevent overfitting
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Visualize the decision tree
plt.figure(figsize=(14, 10))
plot_tree(clf, feature_names=X.columns, class_names=[str(i) for i in np.unique(y)], filled=True, fontsize=8)
plt.show()

# Print the tree in text format
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n")
print(tree_rules)
