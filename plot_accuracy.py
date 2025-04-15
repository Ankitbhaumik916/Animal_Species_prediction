import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import cm

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Normalize dataset (Min-Max Scaling)
def normalize(df_input):
    return (df_input - df_input.min()) / (df_input.max() - df_input.min())

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

# Confusion Matrix
def confusion_matrix(actual, predicted, class_labels):
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for a, p in zip(actual, predicted):
        matrix[class_labels.index(a)][class_labels.index(p)] += 1
    return matrix

# Accuracy
def accuracy_score(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual)

# Heatmap for confusion matrix
def plot_confusion_matrix(matrix, class_labels):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=cm.Blues)
    plt.title("Confusion Matrix")
    fig.colorbar(cax)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_yticklabels(class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            ax.text(j, i, matrix[i][j], va='center', ha='center', color='red')

    plt.tight_layout()
    plt.show()

# PCA from scratch (for 2D projection)
def pca_2d(X):
    mean_vec = np.mean(X, axis=0)
    centered = X - mean_vec
    cov_matrix = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals)[::-1]
    principal_components = eig_vecs[:, sorted_indices[:2]]
    return centered @ principal_components

# KNN visualization function
def visualize_knn(X, y, knn_model, class_mapping, highlight=None):
    X_2d = pca_2d(X)
    labels = np.unique(y)

    plt.figure(figsize=(10, 7))
    for label in labels:
        idxs = y == label
        plt.scatter(X_2d[idxs, 0], X_2d[idxs, 1], label=class_mapping[label], alpha=0.7)

    if highlight is not None:
        x_highlight = highlight.reshape(1, -1)
        x_highlight_norm = (x_highlight - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        x_2d = pca_2d(np.vstack([X, x_highlight_norm]))[-1]
        pred = knn_model.predict(x_highlight_norm)[0]
        species = class_mapping.get(pred, "Unknown")
        plt.scatter(x_2d[0], x_2d[1], c='black', s=200, edgecolors='yellow', label=f'Prediction: {species}', marker='X')

    plt.title("KNN Classification Visualization (2D PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Load and process data
zoo_file = 'zoo.csv'
class_file = 'class.csv'
df = load_data(zoo_file)
class_df = load_data(class_file)

class_mapping = dict(zip(class_df['Class_Number'], class_df['Class_Type']))
X = df.drop(columns=['animal_names', 'class_type'])
y = df['class_type']
X = normalize(X)

# Train and test
knn = KNN(k=3)
knn.fit(X.values, y.values)
predictions = knn.predict(X.values)

# Accuracy and Confusion Matrix
acc = accuracy_score(y.values, predictions)
labels = sorted(y.unique())
cmatrix = confusion_matrix(list(y.values), list(predictions), labels)

print(f"\n✅ Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:")
print(pd.DataFrame(cmatrix, index=labels, columns=labels))

plot_confusion_matrix(cmatrix, class_labels=labels)

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
    print(f"\n🔮 Predicted class for {animal_name}: {species} (Class {prediction})")

    # Visualization with new prediction
    visualize_knn(X.values, y.values, knn, class_mapping, highlight=X_input)

# Run when called directly
if __name__ == "__main__":
    predict_animal()
