# iris_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train kNN model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# UI
st.title("🌼 Iris Flower Classifier")
st.write("Predict the species of an Iris flower based on its measurements!")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(input_data)[0]
    species = iris.target_names[prediction]
    
    st.success(f"**Predicted Species:** {species} ({prediction})")
    st.image(f"https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/iris_images/{species}.jpg", width=200)
