import streamlit as st
import pandas as pd
import pickle
import os

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    model_path = "xgboost_churn_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    return None

model = load_model()

# Header
st.title("Customer Churn Predictor")

if model is None:
    st.error("Error: `xgboost_churn_model.pkl` not found. Please ensure the model file is in the same directory.")
else:
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        country = st.selectbox("Country", options=["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.slider("Age", min_value=18, max_value=100, value=35)
        
    with col2:
        tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
        balance = st.number_input("Balance ($)", min_value=0.0, value=0.0, step=100.0)
        products_number = st.selectbox("Number of Products", options=[1, 2, 3, 4])
        credit_card = st.selectbox("Has Credit Card?", options=["Yes", "No"])
        
    active_member = st.selectbox("Is Active Member?", options=["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=500.0)

    # Conversion logic
    credit_card_val = 1 if credit_card == "Yes" else 0
    active_member_val = 1 if active_member == "Yes" else 0

    # Prepare input data
    input_data = pd.DataFrame([{
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card_val,
        'active_member': active_member_val,
        'estimated_salary': estimated_salary
    }])

    # Predict button
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        st.divider()
        
        if prediction == 1:
            st.error(f"Prediction: **Churn** (Probability: {prediction_proba:.2%})")
            st.warning("The model suggests this customer is likely to leave.")
        else:
            st.success(f"Prediction: **Not Churn** (Probability: {prediction_proba:.2%})")
            st.info("The model suggests this customer is likely to stay.")

# Footer
st.divider()
st.caption("Built with Streamlit")
