import os
import pickle
import pandas as pd
import streamlit as st

MODEL_FILE = "heart_model.pkl"

FEATURES = ["age", "sex", "cp", "thalach"]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC

        df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
        df["target"] = (df["target"] > 0).astype(int)
        X = df[FEATURES].copy().fillna(df[FEATURES].median())
        y = df["target"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        model = SVC()
        model.fit(X_train, y_train)

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        return model

    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("Heart Disease Prediction System")
st.write("Prediction using exactly four input attributes.")

age = st.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4])
thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=50, max_value=250, value=140)

if st.button("Predict"):
    model = load_model()

    input_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")
