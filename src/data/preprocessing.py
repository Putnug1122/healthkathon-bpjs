import pandas as pd
import pickle
from typing import Dict, Any
import networkx as nx
from ..config import Config


class DataPreprocessor:
    def __init__(self):
        self.provider_type_encoder = self._load_encoder(
            "label_encoder_Rndrng_Prvdr_Type.pkl"
        )
        self.hcpcs_encoder = self._load_encoder("label_encoder_HCPCS_Cd.pkl")

    def _load_encoder(self, filename: str):
        with open(Config.ENCODERS_DIR / filename, "rb") as f:
            return pickle.load(f)

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        df = pd.DataFrame([data])

        # Apply encodings
        df["Rndrng_Prvdr_Type_encoded"] = self.provider_type_encoder.transform(
            [data["Rndrng_Prvdr_Type"]]
        )
        df["HCPCS_Cd_encoded"] = self.hcpcs_encoder.transform([data["HCPCS_Cd"]])

        return df
