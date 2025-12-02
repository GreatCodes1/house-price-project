import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("🏡 House Price Prediction App")

model = joblib.load("../models/house_price_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.write("Enter values to predict the house price:")

# Create sample input UI
overall_qual = st.number_input("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
garage_cars = st.number_input("Garage Capacity (Cars)", 0, 5, 1)
total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 4000, 800)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'GarageCars': [garage_cars],
        'TotalBsmtSF': [total_bsmt_sf]
    })

    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]

    st.success(f"Estimated Price: ${pred:,.2f}")
