import streamlit as st
from src.models.predictor import FraudPredictor
from src.data.preprocessing import process_batch_claims
import pandas as pd


def render_single_claim_form():
    """Render the form for single claim processing"""
    with st.form("claim_form"):
        col1, col2 = st.columns(2)

        with col1:
            npi = st.text_input("Provider NPI")
            hcpcs = st.text_input("HCPCS Code")
            provider_type = st.text_input("Provider Type")
            gender = st.selectbox("Provider Gender", ["M", "F"])
            drug_ind = st.selectbox("HCPCS Drug Indicator", ["Y", "N"])
            place_of_service = st.selectbox("Place of Service", ["F", "O"])

        with col2:
            allowed_amt = st.number_input(
                "Average Medicare Allowed Amount", min_value=0.0
            )
            payment_amt = st.number_input(
                "Average Medicare Payment Amount", min_value=0.0
            )
            standardized_amt = st.number_input(
                "Average Medicare Standardized Amount", min_value=0.0
            )
            submitted_charge = st.number_input(
                "Average Submitted Charge", min_value=0.0
            )
            bene_day_services = st.number_input(
                "Total Beneficiary Day Services", min_value=0
            )
            total_benes = st.number_input("Total Beneficiaries", min_value=0)
            total_services = st.number_input("Total Services", min_value=0)

        submitted = st.form_submit_button("Check for Fraud")

        return submitted, {
            "Rndrng_NPI": npi,
            "HCPCS_Cd": hcpcs,
            "Rndrng_Prvdr_Type": provider_type,
            "Rndrng_Prvdr_Gndr": gender,
            "HCPCS_Drug_Ind": drug_ind,
            "Place_Of_Srvc": place_of_service,
            "Avg_Mdcr_Alowd_Amt": allowed_amt,
            "Avg_Mdcr_Pymt_Amt": payment_amt,
            "Avg_Mdcr_Stdzd_Amt": standardized_amt,
            "Avg_Sbmtd_Chrg": submitted_charge,
            "Tot_Bene_Day_Srvcs": bene_day_services,
            "Tot_Benes": total_benes,
            "Tot_Srvcs": total_services,
        }


def display_prediction_results(prediction, probability, features, model):
    """Display prediction results and feature importance"""
    st.subheader("Prediction Results")
    if prediction == 1:
        st.error("⚠️ Potential Fraud Detected")
        st.write(f"Confidence: {probability[1]:.2%}")
    else:
        st.success("✅ No Fraud Detected")
        st.write(f"Confidence: {probability[0]:.2%}")

    # Display feature importance
    st.subheader("Important Factors")
    feature_importance = pd.DataFrame(
        {
            "Feature": features.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    st.bar_chart(feature_importance.set_index("Feature")["Importance"])


def display_batch_results(results_df, error_rows):
    """Display batch processing results and summary statistics"""
    st.subheader("Batch Processing Results")
    st.write(results_df)

    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        "Download Results",
        csv,
        "fraud_detection_results.csv",
        "text/csv",
        key="download-csv",
    )

    if error_rows:
        st.warning(
            f"Failed to process {len(error_rows)} claims. See error log for details."
        )
        error_df = pd.DataFrame(error_rows)
        st.write("Error Log:")
        st.write(error_df)

    # Show summary statistics
    st.subheader("Summary Statistics")
    total_claims = len(results_df)
    fraud_claims = len(results_df[results_df["Prediction"] == "Fraud"])
    error_claims = len(error_rows)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", total_claims)
    col2.metric("Fraud Detected", fraud_claims)
    col3.metric("Processing Errors", error_claims)


def main():
    st.title("Healthcare Claims Fraud Detection")

    predictor = FraudPredictor()

    tab1, tab2 = st.tabs(["Single Claim", "Batch Processing"])

    with tab1:
        st.write("Enter claim details below to check for potential fraud")
        submitted, input_data = render_single_claim_form()

        if submitted:
            try:
                prediction, probability, features = predictor.predict(input_data)
                display_prediction_results(
                    prediction, probability, features, predictor.model
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check your input values and try again.")

    with tab2:
        st.write("Upload a CSV file with multiple claims to process in batch")
        st.write("File should contain the same columns as the single claim form")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.write(df.head())

                if st.button("Process Batch"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process batch claims
                    results_df, error_rows = process_batch_claims(predictor, df)

                    status_text.text("Processing complete!")
                    display_batch_results(results_df, error_rows)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.write(
                    "Please ensure your CSV file has the correct format and column names."
                )


if __name__ == "__main__":
    main()
