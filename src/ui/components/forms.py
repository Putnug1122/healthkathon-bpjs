import streamlit as st
from typing import Optional, Dict, Any


class PredictionForm:
    def render(self) -> Optional[Dict[str, Any]]:
        """Render the prediction form and return the input data if submitted."""
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                npi = st.text_input("Provider NPI", "1234567890")
                hcpcs = st.text_input("HCPCS Code", "99213")
                provider_type = st.selectbox(
                    "Provider Type",
                    ["Family Practice", "Internal Medicine", "Cardiology"],
                )
                gender = st.selectbox("Provider Gender", ["M", "F"])

            with col2:
                avg_charge = st.number_input("Average Submitted Charge", value=100.0)
                total_services = st.number_input("Total Services", value=1)
                total_beneficiaries = st.number_input("Total Beneficiaries", value=1)
                place_of_service = st.selectbox("Place of Service", ["F", "O"])

            submitted = st.form_submit_button("Submit")

            if submitted:
                return {
                    "Rndrng_NPI": npi,
                    "HCPCS_Cd": hcpcs,
                    "Rndrng_Prvdr_Type": provider_type,
                    "Rndrng_Prvdr_Gndr": gender,
                    "Avg_Sbmtd_Chrg": avg_charge,
                    "Tot_Srvcs": total_services,
                    "Tot_Benes": total_beneficiaries,
                    "Place_Of_Srvc": place_of_service,
                    "HCPCS_Drug_Ind": "N",  # Default value
                    "Avg_Mdcr_Alowd_Amt": avg_charge * 0.8,  # Estimated
                    "Avg_Mdcr_Pymt_Amt": avg_charge * 0.7,  # Estimated
                    "Avg_Mdcr_Stdzd_Amt": avg_charge * 0.75,  # Estimated
                    "Tot_Bene_Day_Srvcs": total_services,  # Estimated
                }
        return None
