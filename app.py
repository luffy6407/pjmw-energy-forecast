import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PJMW Energy Forecast",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# LOAD FILES
# =====================================================

model = joblib.load("xgb_energy_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_csv("PJMW_hourly_processed.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime")

df.set_index("Datetime", inplace=True)

# =====================================================
# TITLE
# =====================================================

st.title("⚡ PJMW Energy Consumption Forecast")

st.markdown(
    """
    Forecast future PJMW electricity consumption using the trained
    XGBoost forecasting model.
    """
)

forecast_days = st.selectbox(
    "Select Forecast Horizon",
    [1, 7, 15, 30],
    index=1
)

# =====================================================
# FORECAST FUNCTION
# =====================================================

def forecast_future(days):

    history = df.copy()

    future_predictions = []

    future_hours = days * 24

    for _ in range(future_hours):

        next_time = history.index[-1] + pd.Timedelta(hours=1)

        hour = next_time.hour
        dayofweek = next_time.dayofweek
        month = next_time.month
        quarter = next_time.quarter
        year = next_time.year
        dayofyear = next_time.dayofyear

        lag24 = history["PJMW_MW"].iloc[-24]
        lag48 = history["PJMW_MW"].iloc[-48]
        lag168 = history["PJMW_MW"].iloc[-168]

        rolling24_mean = history["PJMW_MW"].iloc[-24:].mean()
        rolling168_mean = history["PJMW_MW"].iloc[-168:].mean()
        rolling_30day = history["PJMW_MW"].iloc[-720:].mean()

        row = pd.DataFrame({
            "hour": [hour],
            "dayofweek": [dayofweek],
            "month": [month],
            "quarter": [quarter],
            "year": [year],
            "dayofyear": [dayofyear],
            "rolling_30day": [rolling_30day],
            "lag24": [lag24],
            "lag48": [lag48],
            "lag168": [lag168],
            "rolling24_mean": [rolling24_mean],
            "rolling168_mean": [rolling168_mean]
        })

        row = row[feature_columns]

        row_scaled = scaler.transform(row)

        pred = model.predict(row_scaled)[0]

        future_predictions.append(pred)

        history.loc[next_time, "PJMW_MW"] = pred

    forecast_df = pd.DataFrame({
        "Datetime": pd.date_range(
            start=df.index[-1] + pd.Timedelta(hours=1),
            periods=future_hours,
            freq="h"
        ),
        "Forecast_MW": future_predictions
    })

    return forecast_df

# =====================================================
# FORECAST BUTTON
# =====================================================

if st.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        forecast_df = forecast_future(forecast_days)

    st.success(
        f"{forecast_days}-Day Forecast Generated Successfully"
    )

    # ---------------------------------
    # Metrics
    # ---------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Forecast Horizon",
            f"{forecast_days} Days"
        )

    with col2:
        st.metric(
            "Forecast Hours",
            len(forecast_df)
        )

    with col3:
        st.metric(
            "Average Forecast",
            f"{forecast_df['Forecast_MW'].mean():,.0f} MW"
        )

    # ---------------------------------
    # Forecast Table
    # ---------------------------------

    st.subheader("Forecast Data")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # ---------------------------------
    # Historical + Forecast Plot
    # ---------------------------------

    st.subheader("Forecast Visualization")

    historical = df[["PJMW_MW"]].tail(24 * 30)

    historical = historical.rename(
        columns={"PJMW_MW": "Historical"}
    )

    forecast_plot = forecast_df.copy()

    forecast_plot.set_index(
        "Datetime",
        inplace=True
    )

    forecast_plot.rename(
        columns={"Forecast_MW": "Forecast"},
        inplace=True
    )

    combined = pd.concat(
        [historical, forecast_plot],
        axis=1
    )

    st.line_chart(combined)

    # ---------------------------------
    # Download CSV
    # ---------------------------------

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="pjmw_energy_forecast.csv",
        mime="text/csv"
    )