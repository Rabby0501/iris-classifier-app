import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train final model
best_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
best_rf.fit(X, y)

# Streamlit app
st.set_page_config(page_title="Iris Classification", layout="wide")
st.title("🌸 Iris Flower Classification with Random Forest")

# Sidebar for user controls
with st.sidebar:
    st.header("Live Prediction")
    sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_w = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    
    if st.button("Predict"):
        prediction = best_rf.predict([[sepal_l, sepal_w, petal_l, petal_w]])[0]
        st.success(f"Predicted Species: **{iris.target_names[prediction]}**")
        st.image(f"https://raw.githubusercontent.com/your-repo/iris_images/main/{iris.target_names[prediction]}.jpg")

# Main content tabs
tab1, tab2 = st.tabs(["Project Steps", "Technical Details"])

with tab1:
    st.header("Key Project Steps")
    
    with st.expander("1. Data Preparation", expanded=True):
        st.write("**Iris Dataset Overview**")
        df = pd.DataFrame(X, columns=iris.feature_names)
        df['Species'] = y
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Dataset Shape: {X.shape[0]} samples, {X.shape[1]} features")

    with st.expander("2. Data Cleaning"):
        st.write("**Outlier Detection with Boxplots**")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i, col in enumerate(iris.feature_names):
            df.boxplot(column=col, ax=axes[i//2, i%2])
        st.pyplot(fig)

    with st.expander("3. Data Normalization"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.write("**Before Normalization**")
        st.write(pd.DataFrame(X, columns=iris.feature_names).describe().loc[['mean', 'std']])
        st.write("**After Normalization**")
        st.write(pd.DataFrame(X_scaled, columns=iris.feature_names).describe().loc[['mean', 'std']])

    with st.expander("4. Hyperparameter Tuning"):
        params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 3, 5]}
        grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        grid.fit(X_train, y_train)
        st.write("**Best Parameters Found:**")
        st.code(f"{grid.best_params_}")

with tab2:
    st.header("Model Performance")
    
    with st.expander("5. Model Evaluation"):
        y_pred = best_rf.predict(X_test)
        st.write(f"**Accuracy: {accuracy_score(y_test, y_pred):.2%}**")
        fig = plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)

    with st.expander("6. Feature Importance"):
        importance = pd.DataFrame({
            'Feature': iris.feature_names,
            'Importance': best_rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=importance)
        st.pyplot(fig)

    with st.expander("7. Cross-Validation"):
        scores = cross_val_score(best_rf, X, y, cv=5)
        st.write("**Cross-Validation Scores:**")
        st.write(scores)
        st.write(f"**Average Score: {scores.mean():.2%}**")