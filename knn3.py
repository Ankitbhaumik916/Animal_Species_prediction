from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
knn, X_min, X_max, class_mapping = joblib.load("knn_model.pkl")

# Load dataset for reference
zoo_file = "zoo.csv"
df = pd.read_csv(zoo_file)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # JSON input from frontend
    animal_name = data.get("animal_name", "").strip().lower()

    # Check if the animal exists in the dataset
    animal_row = df[df["animal_names"].str.lower() == animal_name]

    if not animal_row.empty:
        X_input = animal_row.drop(columns=["animal_names", "class_type"]).values
    else:
        # If not found, use input features from frontend
        X_input = np.array([data["features"]])

    # Normalize input
    X_input = (X_input - X_min.values) / (X_max.values - X_min.values)

    # Get prediction
    prediction = knn.predict(X_input)[0]
    species = class_mapping.get(prediction, "Unknown")

    return jsonify({"prediction": species, "class_number": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
