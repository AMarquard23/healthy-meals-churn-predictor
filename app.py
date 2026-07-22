import pickle
import pandas as pd
import streamlit as st

# Load saved files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Page setup
st.set_page_config(
    page_title="Healthy Meals Churn Predictor",
    page_icon="🥗"
)

st.title("🥗 Healthy Meals Customer Churn Predictor")
st.write(
    "Enter customer activity and demographic information "
    "to estimate the probability of churn."
)

# Customer activity inputs
st.subheader("Customer Activity")

total_num_sessions = st.number_input(
    "Total Number of Sessions", min_value=0, value=30
)

gross_total_session_length = st.number_input(
    "Gross Total Session Length (Minutes)", min_value=0, value=600
)

active_days = st.number_input(
    "Number of Active Days", min_value=0, value=10
)

active_quarters = st.slider(
    "Number of Active Quarters", 0, 4, 4
)

# Demographic inputs
st.subheader("Customer Demographics")

age = st.slider("Customer Age", 18, 100, 40)
tech_comfort_score = st.slider(
    "Technology Comfort Score", 1, 10, 5
)

education = st.selectbox(
    "Education Level",
    ["Graduate", "High School", "Other", "Post-Graduate"]
)

income_level = st.selectbox(
    "Income Level",
    ["High", "Low", "Medium", "Very High"],
    index=2
)

device_type = st.selectbox(
    "Device Type",
    ["Desktop-only", "Mobile-only", "Multi-device"],
    index=2
)

# Generate prediction
if st.button("Predict Churn"):
    avg_sessions = (
        total_num_sessions / active_quarters
        if active_quarters > 0 else 0
    )

    customer_data = pd.DataFrame({
        "TOTAL_NUM_SESSIONS": [total_num_sessions],
        "GROSS_TOTAL_SESSION_LENGTH": [gross_total_session_length],
        "ACTIVE_DAYS": [active_days],
        "ACTIVE_QUARTERS": [active_quarters],
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": [avg_sessions],
        "AGE": [age],
        "TECH_COMFORT_SCORE": [tech_comfort_score],
        "EDUCATION": [education],
        "INCOME_LEVEL": [income_level],
        "DEVICE_TYPE": [device_type]
    })

    categorical_cols = [
        "EDUCATION",
        "INCOME_LEVEL",
        "DEVICE_TYPE"
    ]

    encoded_df = pd.DataFrame(
        encoder.transform(customer_data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    prediction_df = pd.concat(
        [
            customer_data.drop(columns=categorical_cols),
            encoded_df
        ],
        axis=1
    ).reindex(columns=feature_columns, fill_value=0)

    churn_probability, renewal_probability = (
        model.predict_proba(prediction_df)[0]
    )

    st.subheader("Prediction Results")

    if churn_probability >= 0.50:
        st.error("Predicted Status: High Churn Risk")
    else:
        st.success("Predicted Status: Low Churn Risk")

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{churn_probability:.1%}")
    col2.metric("Renewal Probability", f"{renewal_probability:.1%}")
