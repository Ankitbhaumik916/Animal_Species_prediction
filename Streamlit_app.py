import streamlit as st
import numpy as np
import pandas as pd
from knn2 import KNN, class_mapping, normalize, X, df

# Load trained KNN model
knn = KNN(k=3)
knn.fit(X.values, df['class_type'].values)

st.set_page_config(page_title="Animal Species Predictor", layout="centered")
st.title("🧬 Animal Species Predictor")
st.markdown("Upload animal features to predict the species")

# Feature input via sidebar or form
with st.form("animal_features_form"):
    hair = st.selectbox("Hair", [0, 1])
    feathers = st.selectbox("Feathers", [0, 1])
    eggs = st.selectbox("Lays Eggs", [0, 1])
    milk = st.selectbox("Gives Milk", [0, 1])
    airborne = st.selectbox("Airborne", [0, 1])
    aquatic = st.selectbox("Aquatic", [0, 1])
    predator = st.selectbox("Predator", [0, 1])
    toothed = st.selectbox("Toothed", [0, 1])
    backbone = st.selectbox("Backbone", [0, 1])
    breathes = st.selectbox("Breathes", [0, 1])
    venomous = st.selectbox("Venomous", [0, 1])
    fins = st.selectbox("Fins", [0, 1])
    legs = st.selectbox("Legs", [0, 2, 4, 6, 8])
    tail = st.selectbox("Tail", [0, 1])
    domestic = st.selectbox("Domestic", [0, 1])
    catsize = st.selectbox("Cat Size", [0, 1])

    submitted = st.form_submit_button("Predict")
if submitted:
    input_features = np.array([
        hair, feathers, eggs, milk, airborne, aquatic,
        predator, toothed, backbone, breathes, venomous,
        fins, legs, tail, domestic, catsize
    ]).reshape(1, -1)

    # Normalize input
    normalized_input = normalize(pd.DataFrame(input_features, columns=X.columns)).values

    # Prediction
    predicted_class = knn.predict(normalized_input)[0]
    predicted_animal = class_mapping.get(predicted_class, "Unknown")

    # 🔍 Debug Output
    st.subheader("🧪 Debug Info")
    st.code(f"Normalized Input: {normalized_input.tolist()}")
    st.write(f"Predicted Class ID: {predicted_class}")
    st.write(f"Mapped Species: {predicted_animal}")

    st.success(f"🧠 Predicted Species: **{predicted_animal}**")
