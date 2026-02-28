import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
        }
    </style>
""", unsafe_allow_html=True)
# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="🏦")

# --- CUSTOM CSS ---
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: None;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        text-transform: uppercase;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "xgboost_churn_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    return None

model = load_model()

# --- SIDEBAR INPUTS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80) 
st.sidebar.title("Customer Profile")
st.sidebar.markdown("Add Your Customer Details Here")

with st.sidebar.expander("Demographics", expanded=True):
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    country = st.selectbox("Country", options=["France", "Spain", "Germany"])

with st.sidebar.expander("Financial Info", expanded=True):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)

with st.sidebar.expander("Banking Relationship", expanded=True):
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
    products_number = st.selectbox("Number of Products", options=[1, 2, 3, 4])
    credit_card = st.radio("Has Credit Card?", options=["Yes", "No"], horizontal=True)
    active_member = st.radio("Is Active Member?", options=["Yes", "No"], horizontal=True)

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

# --- MAIN DASHBOARD ---
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Analyze customer retention risk instantly with our advanced XGBoost Machine Learning model.")
st.markdown("---")

# Metrics Summary Row
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f'<div class="metric-card"><div class="metric-label">Credit Score</div><div class="metric-value">{credit_score}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><div class="metric-label">Balance</div><div class="metric-value">${balance:,.0f}</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><div class="metric-label">Salary</div><div class="metric-value">${estimated_salary:,.0f}</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-card"><div class="metric-label">Tenure</div><div class="metric-value">{tenure} Yrs</div></div>', unsafe_allow_html=True)

st.write("")

# Prediction Layout
res_col, chart_col = st.columns([1, 1.5])

with res_col:
    st.subheader("Action")
    st.write("Click below to evaluate the customer's churn risk profile.")
    
    if model is None:
        st.error("Model not found. Ensure `xgboost_churn_model.pkl` is in the directory.")
        predict_clicked = False
    else:
        predict_clicked = st.button("🔮 Predict Churn Risk")

    if predict_clicked:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        st.subheader("Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn")
            st.warning("Immediate intervention recommended. Consider reaching out with personalized offers or checking satisfaction levels.")
        else:
            st.success(f"✅ Customer is Safe")
            st.info("The customer is likely to stay. Continue maintaining good service relations.")

# --- GAUGE CHART ---
with chart_col:
    st.subheader("Probability Analysis")
    if predict_clicked:
        prob_percentage = prediction_proba * 100
        
        # Determine gauge color
        if prob_percentage < 30:
            color = "green"
        elif prob_percentage < 70:
            color = "orange"
        else:
            color = "red"

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)", 'font': {'size': 20, 'color': '#2c3e50'}},
            number = {'suffix': "%", 'font': {'size': 40, 'color': color}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.1)'},
                    {'range': [30, 70], 'color': 'rgba(255, 165, 0, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, b=20, t=50))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Enter details via the sidebar and click 'Predict' to view the full analysis chart.")

st.markdown("---")
st.caption("Built with using Streamlit, Plotly, and scikit-learn.")
