import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("xgb_energy_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("PJMW Energy Consumption Forecast")

hour = st.slider("Hour", 0, 23)

dayofweek = st.slider("Day Of Week", 0, 6)

month = st.slider("Month", 1, 12)

quarter = st.slider("Quarter", 1, 4)

year = st.number_input(
    "Year",
    value=2019
)

dayofyear = st.slider(
    "Day Of Year",
    1,
    366
)

rolling_30day = st.number_input(
    "Rolling 30 Day Mean"
)

lag24 = st.number_input(
    "Lag 24"
)

lag48 = st.number_input(
    "Lag 48"
)

lag168 = st.number_input(
    "Lag 168"
)

rolling24_mean = st.number_input(
    "Rolling 24 Mean"
)

rolling168_mean = st.number_input(
    "Rolling 168 Mean"
)

if st.button("Predict"):

    data = pd.DataFrame({

        'hour':[hour],
        'dayofweek':[dayofweek],
        'month':[month],
        'quarter':[quarter],
        'year':[year],
        'dayofyear':[dayofyear],

        'rolling_30day':[rolling_30day],

        'lag24':[lag24],
        'lag48':[lag48],
        'lag168':[lag168],

        'rolling24_mean':[rolling24_mean],
        'rolling168_mean':[rolling168_mean]

    })

    data_scaled = scaler.transform(data)

    prediction = model.predict(
        data_scaled
    )[0]

    st.success(
        f"Forecasted Energy Consumption: {prediction:.2f} MW"
    )