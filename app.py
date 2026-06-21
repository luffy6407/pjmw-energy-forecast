import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

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

@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly_processed.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    df.set_index("Datetime", inplace=True)
    return df

df = load_data()

model = joblib.load("xgb_energy_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =====================================================
# SIDEBAR
# =====================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Forecast",
        "Historical Analysis",
        "Model Performance",
        "Dataset Summary",
        "About Project"
    ]
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

        row = pd.DataFrame({

            "hour": [next_time.hour],
            "dayofweek": [next_time.dayofweek],
            "month": [next_time.month],
            "quarter": [next_time.quarter],
            "year": [next_time.year],
            "dayofyear": [next_time.dayofyear],

            "rolling_30day": [
                history["PJMW_MW"].iloc[-720:].mean()
            ],

            "lag24": [
                history["PJMW_MW"].iloc[-24]
            ],

            "lag48": [
                history["PJMW_MW"].iloc[-48]
            ],

            "lag168": [
                history["PJMW_MW"].iloc[-168]
            ],

            "rolling24_mean": [
                history["PJMW_MW"].iloc[-24:].mean()
            ],

            "rolling168_mean": [
                history["PJMW_MW"].iloc[-168:].mean()
            ]
        })

        row = row[feature_columns]

        row_scaled = scaler.transform(row)

        pred = model.predict(row_scaled)[0]

        future_predictions.append(pred)

        history.loc[next_time, "PJMW_MW"] = pred

    forecast_df = pd.DataFrame({
        "Datetime": pd.date_range(
            start=pd.Timestamp.now().floor("h"),
            periods=future_hours,
            freq="h"
        ),
        "Forecast_MW": future_predictions
    })

    return forecast_df

# =====================================================
# FORECAST PAGE
# =====================================================

if page == "Forecast":

    st.title("⚡ PJMW Energy Consumption Forecast")

    st.markdown(
        """
        Forecast future PJMW electricity demand using the
        trained XGBoost forecasting model.
        """
    )

    forecast_days = st.slider(
        "Select Forecast Horizon (Days)",
        min_value=1,
        max_value=90,
        value=30
    )

    if st.button("Generate Forecast"):

        with st.spinner("Generating forecast..."):

            forecast_df = forecast_future(forecast_days)

        st.success(
            f"{forecast_days}-Day Forecast Generated Successfully"
        )

        # KPIs

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Forecast Days",
                forecast_days
            )

        with col2:
            st.metric(
                "Forecast Hours",
                len(forecast_df)
            )

        with col3:
            st.metric(
                "Average Load",
                f"{forecast_df['Forecast_MW'].mean():,.0f} MW"
            )

        with col4:
            st.metric(
                "Peak Load",
                f"{forecast_df['Forecast_MW'].max():,.0f} MW"
            )

        # Forecast Insights

        st.subheader("Forecast Insights")

        st.info(
            f"""
            Peak Demand: {forecast_df['Forecast_MW'].max():,.0f} MW

            Lowest Demand: {forecast_df['Forecast_MW'].min():,.0f} MW

            Average Demand: {forecast_df['Forecast_MW'].mean():,.0f} MW
            """
        )

        # Plot

        st.subheader("Forecast Visualization")

        historical = df.tail(24 * 30)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=historical.index,
                y=historical["PJMW_MW"],
                name="Historical"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["Datetime"],
                y=forecast_df["Forecast_MW"],
                name="Forecast"
            )
        )

        fig.update_layout(
            height=600,
            xaxis_title="Date",
            yaxis_title="Energy (MW)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Table

        st.subheader("Forecast Data")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        # Download

        csv = forecast_df.to_csv(index=False)

        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "pjmw_forecast.csv",
            "text/csv"
        )

# =====================================================
# HISTORICAL ANALYSIS
# =====================================================

elif page == "Historical Analysis":

    st.title("📈 Historical Analysis")

    st.subheader("Consumption Patterns")

    col1, col2, col3 = st.columns(3)

    # Hourly Pattern
    with col1:

        hourly_avg = (
            df.groupby(df.index.hour)["PJMW_MW"]
            .mean()
        )

        st.markdown("### Hourly Pattern")

        st.bar_chart(hourly_avg)

    # Day-wise Pattern
    with col2:

        day_avg = (
            df.groupby(df.index.dayofweek)["PJMW_MW"]
            .mean()
        )

        day_avg.index = [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun"
        ]

        st.markdown("### Day-wise Pattern")

        st.bar_chart(day_avg)

    # Monthly Pattern
    with col3:

        monthly_avg = (
            df.groupby(df.index.month)["PJMW_MW"]
            .mean()
        )

        st.markdown("### Monthly Pattern")

        st.line_chart(monthly_avg)

    st.subheader("Historical Demand Trend")

    st.line_chart(
        df["PJMW_MW"].tail(2000)
    )

    st.subheader("Yearly Energy Trend")

    yearly = (
        df["PJMW_MW"]
        .resample("Y")
        .mean()
    )

    st.line_chart(yearly)

    st.subheader("30-Day Rolling Average")

    rolling_avg = (
        df["PJMW_MW"]
        .rolling(24 * 30)
        .mean()
    )

    st.line_chart(
        rolling_avg.tail(3000)
    )

    st.subheader("Demand Distribution")

    fig = px.histogram(
        df,
        x="PJMW_MW",
        nbins=50,
        title="Distribution of Energy Demand"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    peak_hours = df[
        df.index.hour.isin(
            range(8, 21)
        )
    ]["PJMW_MW"]

    off_peak = df[
        ~df.index.hour.isin(
            range(8, 21)
        )
    ]["PJMW_MW"]

    comparison = pd.DataFrame({

        "Category": [
            "Peak Hours",
            "Off Peak Hours"
        ],

        "Average MW": [
            peak_hours.mean(),
            off_peak.mean()
        ]
    })

    st.subheader(
        "Peak vs Off-Peak Consumption"
    )

    st.bar_chart(
        comparison.set_index(
            "Category"
        )
    )
# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("🏆 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "209.60 MW")
    col2.metric("RMSE", "276.88 MW")
    col3.metric("R² Score", "0.9225")

    ranking = pd.DataFrame({

        "Model": [
            "XGBoost",
            "Random Forest",
            "Linear Regression",
            "Holt-Winters",
            "FB Prophet"
        ],

        "R² Score": [
            0.9225,
            0.9005,
            0.8606,
            -0.0446,
            -0.6424
        ]
    })

    st.subheader("Model Ranking")

    st.dataframe(
        ranking,
        use_container_width=True
    )

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )

# =====================================================
# DATASET SUMMARY
# =====================================================

elif page == "Dataset Summary":

    st.title("📊 Dataset Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Start Date",
        str(df.index.min().date())
    )

    col3.metric(
        "End Date",
        str(df.index.max().date())
    )

    col4.metric(
    "Average Demand",
    f"{df['PJMW_MW'].mean():,.0f} MW"
   )

    col5.metric(
    "Maximum Demand",
    f"{df['PJMW_MW'].max():,.0f} MW"
  )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )
    
    # =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.markdown("""

    ## PJMW Energy Consumption Forecasting

    ### Objective

    Forecast future electricity demand using
    machine learning techniques.

    ### Best Model

    XGBoost Regressor

    ### Performance Metrics

    - MAE : 209.60 MW
    - RMSE : 276.88 MW
    - R² Score : 0.9225

    ### Features Used

    - Hour
    - Day Of Week
    - Month
    - Quarter
    - Year
    - Day Of Year
    - Lag24
    - Lag48
    - Lag168
    - Rolling24 Mean
    - Rolling168 Mean
    - Rolling30 Day Mean

    ### Technologies

    - Python
    - Pandas
    - Scikit-Learn
    - XGBoost
    - Streamlit
    - Plotly

    """)