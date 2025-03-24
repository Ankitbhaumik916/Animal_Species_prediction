import numpy as np
import pandas as pd
from collections import Counter
import joblib

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Normalize dataset (Min-Max Scaling)
def normalize(X, X_min=None, X_max=None):
    if X_min is None or X_max is None:
        X_min, X_max = X.min(), X.max()
    return (X - X_min) / (X_max - X_min), X_min, X_max

# Compute Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Algorithm
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Load datasets
zoo_file = 'zoo.csv'
class_file = 'class.csv'
df = load_data(zoo_file)
class_df = load_data(class_file)

# Create class mapping
class_mapping = dict(zip(class_df['Class_Number'], class_df['Class_Type']))

# Extract features and labels
X = df.drop(columns=['animal_names', 'class_type'])  # Features
y = df['class_type']  # Labels

# Normalize features
X, X_min, X_max = normalize(X)

# Train-test split manually
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size].values, X[train_size:].values
y_train, y_test = y[:train_size].values, y[train_size:].values

# Train KNN with optimal K
best_k = 3  # Adjust as needed
knn = KNN(k=best_k)
knn.fit(X_train, y_train)

# Save the model
joblib.dump((knn, X_min, X_max, class_mapping), "knn_model.pkl")
print("Model saved as knn_model.pkl")
