import streamlit as st
import pandas as pd
import requests
from io import StringIO


def show():
    st.title("Batch Prediction")
    st.write("Upload a CSV file with multiple claims for batch prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            with st.spinner("Processing claims..."):
                predictions = []
                for _, row in df.iterrows():
                    try:
                        response = requests.post(
                            "http://localhost:8000/predict", json=row.to_dict()
                        )
                        result = response.json()
                        predictions.append(result)
                    except Exception as e:
                        st.error(f"Error processing row: {str(e)}")
                        continue

                results_df = pd.DataFrame(predictions)

                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results_df)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv",
                    key="download-csv",
                )
