import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction App")
st.write("Provide the house details below to estimate its market price.")

# Load model and scaler
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.subheader("Enter House Details")

OverallQual = st.number_input("Overall Quality (1–10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
GarageCars = st.number_input("Garage Capacity (Cars)", 0, 5, 1)
GarageArea = st.number_input("Garage Area (sq ft)", 0, 2000, 500)
TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 4000, 800)
FullBath = st.number_input("Number of Full Bathrooms", 0, 5, 2)
YearBuilt = st.number_input("Year Built", 1800, 2024, 2000)
LotArea = st.number_input("Lot Area (sq ft)", 500, 50000, 10000)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'OverallQual': [OverallQual],
        'GrLivArea': [GrLivArea],
        'GarageCars': [GarageCars],
        'GarageArea': [GarageArea],
        'TotalBsmtSF': [TotalBsmtSF],
        'FullBath': [FullBath],
        'YearBuilt': [YearBuilt],
        'LotArea': [LotArea]
    })

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]

    st.success(f"💰 Estimated House Price: ${pred:,.2f}")
