import pickle
import pandas as pd
from typing import Dict, Tuple, Any


class FraudPredictor:
    def __init__(self):
        self.load_encoders()
        self.load_model()
        self.load_graph_data()

    def load_encoders(self):
        """Load the Label Encoders for HCPCS_Cd and Rndrng_Prvdr_Type"""
        with open("artifacts/label_encoders/label_encoder_HCPCS_Cd.pkl", "rb") as f:
            self.hcpcs_encoder = pickle.load(f)
        with open(
            "artifacts/label_encoders/label_encoder_Rndrng_Prvdr_Type.pkl", "rb"
        ) as f:
            self.provider_type_encoder = pickle.load(f)

        # Store the known categories
        self.known_hcpcs_codes = set(self.hcpcs_encoder.classes_)
        self.known_provider_types = set(self.provider_type_encoder.classes_)

    def load_model(self):
        """Load the XGBoost model"""
        with open("artifacts/models/xgb_model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def load_graph_data(self):
        """Load pre-computed graph centrality measures"""
        self.centrality_data = pd.read_csv("artifacts/graphs/centrality_measures.csv")
        self.centrality_data.set_index("Rndrng_NPI", inplace=True)

    def get_centrality_measures(self, npi: str) -> Dict[str, float]:
        """Get pre-computed centrality measures for the given NPI"""
        if npi in self.centrality_data.index:
            return self.centrality_data.loc[npi].to_dict()
        return self.centrality_data.median().to_dict()

    def encode_categorical(self, input_data: Dict[str, Any]) -> Tuple[int, int]:
        """Encode categorical variables"""
        hcpcs = input_data["HCPCS_Cd"]
        provider_type = input_data["Rndrng_Prvdr_Type"]

        # Handle unknown categories
        if isinstance(hcpcs, str) and hcpcs not in self.known_hcpcs_codes:
            hcpcs = self.hcpcs_encoder.classes_[0]

        if (
            isinstance(provider_type, str)
            and provider_type not in self.known_provider_types
        ):
            provider_type = self.provider_type_encoder.classes_[0]

        # Encode HCPCS_Cd and Rndrng_Prvdr_Type
        hcpcs_encoded = self.hcpcs_encoder.transform([hcpcs])[0]
        provider_type_encoded = self.provider_type_encoder.transform([provider_type])[0]

        # Binary encodings
        binary_encodings = {
            "Rndrng_Prvdr_Gndr": {"M": 1, "F": 0},
            "HCPCS_Drug_Ind": {"Y": 1, "N": 0},
            "Place_Of_Srvc": {"F": 1, "O": 0},
        }

        for field, mapping in binary_encodings.items():
            input_data[field] = mapping[input_data[field]]

        return hcpcs_encoded, provider_type_encoded

    def predict(
        self, input_data: Dict[str, Any]
    ) -> Tuple[int, Tuple[float, float], pd.DataFrame]:
        """Process a single claim and return prediction"""
        # Encode categorical variables
        self.encode_categorical(input_data)

        # Get centrality measures
        centrality_measures = self.get_centrality_measures(input_data["Rndrng_NPI"])

        # Prepare features for model
        features = pd.DataFrame(
            {
                "Avg_Mdcr_Alowd_Amt": [float(input_data["Avg_Mdcr_Alowd_Amt"])],
                "Avg_Mdcr_Pymt_Amt": [float(input_data["Avg_Mdcr_Pymt_Amt"])],
                "Avg_Mdcr_Stdzd_Amt": [float(input_data["Avg_Mdcr_Stdzd_Amt"])],
                "Avg_Sbmtd_Chrg": [float(input_data["Avg_Sbmtd_Chrg"])],
                "Tot_Bene_Day_Srvcs": [float(input_data["Tot_Bene_Day_Srvcs"])],
                "Tot_Benes": [float(input_data["Tot_Benes"])],
                "Tot_Srvcs": [float(input_data["Tot_Srvcs"])],
                "Rndrng_Prvdr_Gndr": [input_data["Rndrng_Prvdr_Gndr"]],
                "HCPCS_Drug_Ind": [input_data["HCPCS_Drug_Ind"]],
                "Place_Of_Srvc": [input_data["Place_Of_Srvc"]],
                **{k: [v] for k, v in centrality_measures.items()},
            }
        )

        # Make prediction
        prediction = self.model.predict(features)
        probability = self.model.predict_proba(features)

        return prediction[0], probability[0], features
