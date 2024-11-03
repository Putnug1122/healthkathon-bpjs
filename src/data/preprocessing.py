# import pandas as pd
# import pickle
# from typing import Dict, Any
# import networkx as nx
# from ..config import Config


# class DataPreprocessor:
#     def __init__(self):
#         self.provider_type_encoder = self._load_encoder(
#             "label_encoder_Rndrng_Prvdr_Type.pkl"
#         )
#         self.hcpcs_encoder = self._load_encoder("label_encoder_HCPCS_Cd.pkl")

#         # Add binary encoding mappings
#         self.binary_encodings = {
#             "HCPCS_Drug_Ind": {"Y": 1, "N": 0},
#             "Place_Of_Srvc": {"F": 1, "O": 0},
#             "Rndrng_Prvdr_Gndr": {"M": 1, "F": 0},
#         }

#     def _load_encoder(self, filename: str):
#         with open(Config.ENCODERS_DIR / filename, "rb") as f:
#             return pickle.load(f)

#     def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
#         """Preprocess input data for prediction."""
#         df = pd.DataFrame([data])

#         # Apply label encodings
#         df["Rndrng_Prvdr_Type_encoded"] = self.provider_type_encoder.transform(
#             [data["Rndrng_Prvdr_Type"]]
#         )
#         df["HCPCS_Cd_encoded"] = self.hcpcs_encoder.transform([data["HCPCS_Cd"]])

#         # Apply binary encodings
#         for column, mapping in self.binary_encodings.items():
#             df[f"{column}_encoded"] = df[column].map(mapping)

#         return df
import pandas as pd
import pickle
from typing import Dict, Any
import networkx as nx
from ..config import Config
import xgboost as xgb


class DataPreprocessor:
    def __init__(self):
        self.provider_type_encoder = self._load_encoder(
            "label_encoder_Rndrng_Prvdr_Type.pkl"
        )
        self.hcpcs_encoder = self._load_encoder("label_encoder_HCPCS_Cd.pkl")

        # Add binary encoding mappings
        self.binary_encodings = {
            "HCPCS_Drug_Ind": {"Y": 1, "N": 0},
            "Place_Of_Srvc": {"F": 1, "O": 0},
            "Rndrng_Prvdr_Gndr": {"M": 1, "F": 0},
        }

    def _load_encoder(self, filename: str):
        with open(Config.ENCODERS_DIR / filename, "rb") as f:
            return pickle.load(f)

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        df = pd.DataFrame([data])

        # Apply label encodings
        df["Rndrng_Prvdr_Type_encoded"] = (
            self.provider_type_encoder.transform([data["Rndrng_Prvdr_Type"]]).astype(
                "int32"
            )  # Convert to int32 for XGBoost
        )
        df["HCPCS_Cd_encoded"] = (
            self.hcpcs_encoder.transform([data["HCPCS_Cd"]]).astype(
                "int32"
            )  # Convert to int32 for XGBoost
        )

        # Apply binary encodings and convert to numeric
        for column, mapping in self.binary_encodings.items():
            df[f"{column}_encoded"] = (
                df[column].map(mapping).astype("int32")  # Convert to int32 for XGBoost
            )

        # Ensure all numeric columns are float32 or int32
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_columns:
            if df[col].dtype == "float64":
                df[col] = df[col].astype("float32")
            elif df[col].dtype == "int64":
                df[col] = df[col].astype("int32")

        return df

    def create_dmatrix(self, df: pd.DataFrame) -> xgb.DMatrix:
        """Create DMatrix with proper handling of categorical features."""
        # Ensure we're only using the feature columns
        feature_df = df[Config.FEATURE_COLUMNS].copy()

        # Convert all object/string columns to category type
        for col in feature_df.select_dtypes(include=["object"]).columns:
            feature_df[col] = feature_df[col].astype("category")

        # Create DMatrix with categorical feature handling
        return xgb.DMatrix(
            feature_df,
            enable_categorical=True,  # Enable categorical feature support
        )
