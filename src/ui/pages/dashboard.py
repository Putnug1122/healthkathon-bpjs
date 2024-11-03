import streamlit as st
import pandas as pd
import plotly.express as px
from components.plots import plot_historical_trends


def show():
    st.title("Fraud Detection Dashboard")

    # Load historical data
    try:
        historical_data = pd.read_csv("artifacts/historical_predictions.csv")

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            fraud_rate = len(historical_data[historical_data["prediction"] == 1]) / len(
                historical_data
            )
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")

        with col2:
            avg_amount = historical_data["Avg_Sbmtd_Chrg"].mean()
            st.metric("Avg Claim Amount", f"${avg_amount:,.2f}")

        with col3:
            total_saved = historical_data[historical_data["prediction"] == 1][
                "Avg_Sbmtd_Chrg"
            ].sum()
            st.metric("Potential Savings", f"${total_saved:,.2f}")

        # Trend analysis
        st.subheader("Historical Trends")
        plot_historical_trends(historical_data)

        # Provider analysis
        st.subheader("Provider Type Analysis")
        provider_stats = (
            historical_data.groupby("Rndrng_Prvdr_Type")
            .agg({"prediction": "mean", "Avg_Sbmtd_Chrg": "mean"})
            .reset_index()
        )

        fig = px.scatter(
            provider_stats,
            x="Avg_Sbmtd_Chrg",
            y="prediction",
            size="prediction",
            hover_data=["Rndrng_Prvdr_Type"],
            title="Provider Type Risk Analysis",
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
