from typing import Dict, List, Tuple
import pandas as pd


def process_batch_claims(
    predictor, df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Process multiple claims from a DataFrame"""
    results = []
    error_rows = []

    for idx, row in df.iterrows():
        try:
            input_data = row.to_dict()
            input_data["HCPCS_Cd"] = str(input_data["HCPCS_Cd"])
            input_data["Rndrng_Prvdr_Type"] = str(input_data["Rndrng_Prvdr_Type"])

            pred, prob, _ = predictor.predict(input_data)
            results.append(
                {
                    "Rndrng_NPI": row["Rndrng_NPI"],
                    "HCPCS_Cd": row["HCPCS_Cd"],
                    "Prediction": "Fraud" if pred == 1 else "No Fraud",
                    "Confidence": prob[1] if pred == 1 else prob[0],
                }
            )
        except Exception as e:
            error_rows.append(
                {
                    "Rndrng_NPI": row["Rndrng_NPI"],
                    "HCPCS_Cd": row["HCPCS_Cd"],
                    "Error": str(e),
                }
            )

    return pd.DataFrame(results), error_rows
