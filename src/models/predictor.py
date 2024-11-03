import xgboost as xgb
import pandas as pd
from typing import Dict, Any
from ..config import Config
from ..data.preprocessing import DataPreprocessor
from .graph_features import GraphFeatureCalculator


class FraudPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(Config.MODEL_PATH))
        self.preprocessor = DataPreprocessor()
        self.graph_calculator = GraphFeatureCalculator()

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make fraud prediction for a single claim."""
        # Preprocess input data
        df = self.preprocessor.preprocess_input(data)

        # Calculate graph features
        centrality_measures = self.graph_calculator.calculate_centrality_measures(
            data["Rndrng_NPI"], data
        )

        # Add centrality measures to dataframe
        for key, value in centrality_measures.items():
            df[key] = value

        # Make prediction
        prediction = self.model.predict_proba(df[Config.FEATURE_COLUMNS])[0]

        return {
            "prediction": int(prediction[1] > 0.5),
            "fraud_probability": float(prediction[1]),
            "features_importance": self._get_feature_importance(df),
        }

    def _get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for the prediction."""
        importance = self.model.feature_importances_
        return dict(zip(Config.FEATURE_COLUMNS, importance))
