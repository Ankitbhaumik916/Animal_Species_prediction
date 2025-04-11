import numpy as np
import pandas as pd
from collections import Counter

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Normalize dataset (Min-Max Scaling)
def normalize(df_input):
    return (df_input - X.min()) / (X.max() - X.min())


# Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Class
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# Load and process data
zoo_file = 'zoo.csv'
class_file = 'class.csv'
df = load_data(zoo_file)
class_df = load_data(class_file)

class_mapping = dict(zip(class_df['Class_Number'], class_df['Class_Type']))
X = df.drop(columns=['animal_names', 'class_type'])
y = df['class_type']
X = normalize(X)

# Train model
knn = KNN(k=3)
knn.fit(X.values, y.values)

# Optional CLI prediction for testing
def predict_animal():
    animal_name = input("\nEnter an animal name: ").strip().lower()
    animal_row = df[df['animal_names'].str.lower() == animal_name]
    
    if not animal_row.empty:
        print(f"Features found for {animal_name}, making a prediction...")
        X_input = animal_row.drop(columns=['animal_names', 'class_type']).values
    else:
        print(f"{animal_name} not found. Please enter feature values manually:")
        feature_values = []
        for col in X.columns:
            value = float(input(f"Enter value for {col} (0 or 1): "))
            feature_values.append(value)
        X_input = np.array([feature_values])

    X_input = (X_input - X.min().values) / (X.max().values - X.min().values)
    prediction = knn.predict(X_input)[0]
    species = class_mapping.get(prediction, "Unknown")
    print(f"\nPredicted class for {animal_name}: {species} (Class {prediction})")

# Only run this when directly running the file
if __name__ == "__main__":
    predict_animal()
