import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-header{
    text-align:center;
    color:#1E88E5;
}

.metric-box{
    background:#f8f9fa;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD FILES
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "ann_churn_model.h5")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)

encoders = joblib.load(
    os.path.join(BASE_DIR, "encoders.pkl")
)

features = joblib.load(
    os.path.join(BASE_DIR, "features.pkl")
)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    "<h1 class='main-header'>📡 AI-Powered Customer Churn Prediction Platform</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "### Predict whether a telecom customer is likely to leave in the next month."
)

st.divider()

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("📋 Customer Information")

input_data = {}

for feature in features:

    if feature == "gender":

        value = st.sidebar.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        input_data[feature] = (
            1 if value == "Male" else 0
        )

    elif feature == "SeniorCitizen":

        value = st.sidebar.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        input_data[feature] = (
            1 if value == "Yes" else 0
        )

    elif feature in [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]:

        value = st.sidebar.selectbox(
            feature,
            ["No", "Yes"]
        )

        input_data[feature] = (
            1 if value == "Yes" else 0
        )

    elif feature == "tenure":

        input_data[feature] = st.sidebar.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

    elif feature == "MonthlyCharges":

        input_data[feature] = st.sidebar.slider(
            "Monthly Charges",
            0,
            200,
            70
        )

    elif feature == "TotalCharges":

        input_data[feature] = st.sidebar.number_input(
            "Total Charges",
            value=1000.0
        )

    elif feature == "InternetService":

        option = st.sidebar.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        mapping = {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }

        input_data[feature] = mapping[option]

    elif feature == "Contract":

        option = st.sidebar.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        mapping = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }

        input_data[feature] = mapping[option]

    elif feature == "PaymentMethod":

        option = st.sidebar.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )

        mapping = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer": 2,
            "Credit card": 3
        }

        input_data[feature] = mapping[option]

    else:

        input_data[feature] = st.sidebar.number_input(
            feature,
            value=0.0
        )

# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🚀 Predict Churn",
    use_container_width=True
):

    input_df = pd.DataFrame([input_data])

    scaled_input = scaler.transform(input_df)

    probability = model.predict(
        scaled_input,
        verbose=0
    )[0][0]

    churn_probability = probability * 100

    prediction = (
        "Likely to Churn"
        if probability > 0.5
        else "Likely to Stay"
    )

    # ======================================
    # RISK LEVEL
    # ======================================

    if churn_probability > 80:

        risk = "🔴 CRITICAL"

    elif churn_probability > 60:

        risk = "🟠 HIGH"

    elif churn_probability > 40:

        risk = "🟡 MEDIUM"

    else:

        risk = "🟢 LOW"

    # ======================================
    # DASHBOARD
    # ======================================

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        st.metric(
            "Prediction",
            prediction
        )

    st.progress(
        float(churn_probability / 100)
    )

    st.divider()

    # ======================================
    # RECOMMENDATIONS
    # ======================================

    st.subheader(
        "💡 Retention Recommendations"
    )

    if churn_probability > 80:

        st.error(
            "Immediate retention action required."
        )

        st.info(
            "Offer special discounts and assign dedicated support."
        )

    elif churn_probability > 60:

        st.warning(
            "Customer shows high churn risk."
        )

        st.info(
            "Offer loyalty rewards and upgraded plans."
        )

    else:

        st.success(
            "Customer appears stable."
        )

        st.info(
            "Continue regular engagement campaigns."
        )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-Powered Telecom Customer Churn Prediction using Artificial Neural Networks (ANN)"
)
