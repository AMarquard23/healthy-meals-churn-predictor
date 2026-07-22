import pickle
import pandas as pd
import streamlit as st

# Load saved model files
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# Page settings
st.set_page_config(
    page_title="Healthy Meals Churn Predictor",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 Healthy Meals Customer Churn Predictor")

st.write(
    """
    Enter customer activity and demographic information
    to estimate the probability that a customer will churn.
    """
)
