import streamlit as st
import requests
import pandas as pd
from components.forms import PredictionForm
from components.plots import plot_feature_importance


def show():
    st.title("Medicare Fraud Detection")
    st.write("Enter claim details to check for potential fraud")

    form = PredictionForm()
    data = form.render()

    if data:
        with st.spinner("Making prediction..."):
            try:
                response = requests.post("http://localhost:8000/predict", json=data)
                result = response.json()

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Prediction Results")
                    fraud_prob = result["fraud_probability"]
                    st.metric(
                        "Fraud Probability",
                        f"{fraud_prob:.2%}",
                        delta="High Risk" if fraud_prob > 0.7 else "Low Risk",
                    )

                    if fraud_prob > 0.7:
                        st.warning("⚠️ This claim has been flagged for review")
                    else:
                        st.success("✅ This claim appears legitimate")

                with col2:
                    st.subheader("Feature Importance")
                    plot_feature_importance(result["features_importance"])

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
