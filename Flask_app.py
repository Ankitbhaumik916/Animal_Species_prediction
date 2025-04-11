from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from knn2 import KNN, class_mapping, normalize, X, df  # Import your KNN model

app = Flask(__name__)

# Load trained KNN model
knn = KNN(k=3)
knn.fit(X.values, df['class_type'].values)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Read JSON data
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        # Convert input data to array format
        input_features = np.array([
            int(data["hair"]), int(data["feathers"]), int(data["eggs"]),
            int(data["milk"]), int(data["airborne"]), int(data["aquatic"]),
            int(data["predator"]), int(data["toothed"]), int(data["backbone"]),
            int(data["breathes"]), int(data["venomous"]), int(data["fins"]),
            int(data["legs"]), int(data["tail"]), int(data["domestic"]),
            int(data["catsize"])
        ]).reshape(1, -1)

        # Normalize the input
        normalized_input = normalize(pd.DataFrame(input_features, columns=X.columns)).values

        # Predict the class type
        predicted_class = knn.predict(normalized_input)[0]
        predicted_animal = class_mapping[predicted_class]

        return jsonify({"prediction": predicted_animal})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
