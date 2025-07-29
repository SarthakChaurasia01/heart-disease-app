import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Title
st.title("❤️ Heart Disease Prediction App")
st.write("This app uses machine learning to predict the presence of heart disease based on user input.")

# Load dataset
def load_data():
    return pd.read_csv("heart-disease.csv")

df = load_data()

# Sidebar form for input features
st.sidebar.header("📝 Fill Patient Details")

def user_input_features():
    with st.sidebar.form("heart_form"):
        age = st.slider('Age', 29, 77, 54)
        sex = st.selectbox('Sex (0 = female, 1 = male)', [0, 1])
        cp = st.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3])
        trestbps = st.slider('Resting Blood Pressure', 94, 200, 130)
        chol = st.slider('Serum Cholesterol (mg/dl)', 126, 564, 246)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [0, 1])
        restecg = st.selectbox('Resting ECG Results (0–2)', [0, 1, 2])
        thalach = st.slider('Max Heart Rate Achieved', 71, 202, 150)
        exang = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [0, 1])
        oldpeak = st.slider('ST Depression Induced', 0.0, 6.2, 1.0)
        slope = st.selectbox('Slope of Peak Exercise ST Segment (0–2)', [0, 1, 2])
        ca = st.selectbox('Number of Major Vessels Colored by Flourosopy (0–4)', [0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia (0 = unknown, 1 = normal, 2 = fixed defect, 3 = reversible defect)', [0, 1, 2, 3])

        submit = st.form_submit_button("Predict")

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
    return pd.DataFrame(data, index=[0]), submit

input_df, submit = user_input_features()

# Split and train model
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# If user clicks submit, run prediction
if submit:
    prediction = model.predict(input_df)
    pred_prob = model.predict_proba(input_df)

    st.subheader("🧠 Prediction Result")
    result = '💔 Positive for Heart Disease' if prediction[0] == 1 else '❤️ No Heart Disease'
    st.success(result)

    st.subheader("📊 Prediction Probability")
    st.write(f"Probability of Heart Disease: {pred_prob[0][1]*100:.2f}%")

    st.subheader("📈 Model Performance on Test Data")
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
