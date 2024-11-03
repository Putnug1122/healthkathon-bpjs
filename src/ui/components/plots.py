import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict


def plot_feature_importance(importance_dict: Dict[str, float]):
    """Plot feature importance using Plotly."""
    df = pd.DataFrame(
        {
            "Feature": list(importance_dict.keys()),
            "Importance": list(importance_dict.values()),
        }
    ).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(x=df["Importance"], y=df["Feature"], orientation="h"))

    fig.update_layout(
        title="Feature Importance", xaxis_title="Importance Score", height=400
    )

    st.plotly_chart(fig)


def plot_historical_trends(data: pd.DataFrame):
    """Plot historical fraud detection trends."""
    # Daily fraud rate trend
    daily_fraud = (
        data.groupby(pd.to_datetime(data["timestamp"]).dt.date)["prediction"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        daily_fraud, x="timestamp", y="prediction", title="Daily Fraud Detection Rate"
    )

    st.plotly_chart(fig)
