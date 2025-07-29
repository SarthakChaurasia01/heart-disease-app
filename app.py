import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Title
st.title("❤️ Heart Disease Prediction App")
st.write("This app uses machine learning to predict the presence of heart disease.")

# Load dataset
def load_data():
    return pd.read_csv("heart-disease.csv")

df = load_data()

# Sidebar for input features
st.sidebar.header("Patient Data Input")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex (0 = female, 1 = male)', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG Results (0–2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment (0–2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy (0–4)', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Split and train model
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(input_df)
pred_prob = model.predict_proba(input_df)

st.subheader("Prediction Result")
result = 'Positive for Heart Disease 💔' if prediction[0] == 1 else 'No Heart Disease ❤️'
st.write(result)

st.subheader("Prediction Probability")
st.write(f"Probability of Heart Disease: {pred_prob[0][1]*100:.2f}%")

# Optional: show metrics
st.subheader("Model Performance on Test Data")
y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))
