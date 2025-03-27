from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from knn2 import KNN, class_mapping, normalize, X, df  # Import your model

app = Flask(__name__)

# Load trained KNN model
knn = KNN(k=3)  # Ensure your model is trained before using it
knn.fit(X.values, df['class_type'].values)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    animal_name = data.get("animal_name", "").strip().lower()
    features_input = data.get("features", "").strip()

    if animal_name:
        # Find the animal in the dataset
        animal_row = df[df['animal_names'].str.lower() == animal_name]
        if not animal_row.empty:
            X_input = animal_row.drop(columns=['animal_names', 'class_type']).values
        else:
            return jsonify({"prediction": "Animal not found. Try entering manually."})
    elif features_input:
        try:
            X_input = np.array([float(x) for x in features_input.split(",")]).reshape(1, -1)
        except ValueError:
            return jsonify({"prediction": "Invalid feature values. Enter numbers separated by commas."})
    else:
        return jsonify({"prediction": "No input provided. Please enter an animal name or feature values."})

    # Normalize the input features
    X_input = (X_input - X.min().values) / (X.max().values - X.min().values)

    # Make the prediction
    prediction = knn.predict(X_input)[0]
    species = class_mapping.get(prediction, "Unknown")

    return jsonify({"prediction": species})

if __name__ == "__main__":
    app.run(debug=False)
