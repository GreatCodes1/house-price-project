import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction App")
st.write("Provide the house details below to estimate its market price.")

# ===========================
# LOAD MODEL & SCALER
# ===========================
try:
    model = joblib.load("models/house_price_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.error("Model files not found. Make sure the folder structure is correct:\n"
             "`project/models/house_price_model.pkl`\n"
             "`project/models/scaler.pkl`")
    st.stop()

# ===========================
# INPUT FIELDS
# ===========================
st.subheader("Enter House Details")

overall_qual = st.number_input("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
garage_cars = st.number_input("Garage Capacity (Cars)", 0, 5, 1)
total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 4000, 800)

# ===========================
# PREDICTION
# ===========================
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'GarageCars': [garage_cars],
        'TotalBsmtSF': [total_bsmt_sf]
    })

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 **Estimated House Price:** ${prediction:,.2f}")
