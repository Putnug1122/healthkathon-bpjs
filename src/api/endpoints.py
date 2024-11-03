from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from ..models.predictor import FraudPredictor

app = FastAPI(title="Medicare Fraud Detection API")
predictor = FraudPredictor()


class PredictionRequest(BaseModel):
    Rndrng_NPI: str
    HCPCS_Cd: str
    Rndrng_Prvdr_Type: str
    Avg_Mdcr_Alowd_Amt: float
    Avg_Mdcr_Pymt_Amt: float
    Avg_Mdcr_Stdzd_Amt: float
    Avg_Sbmtd_Chrg: float
    Tot_Bene_Day_Srvcs: int
    Tot_Benes: int
    Tot_Srvcs: int
    Rndrng_Prvdr_Gndr: str
    HCPCS_Drug_Ind: str
    Place_Of_Srvc: str


@app.post("/predict")
async def predict_fraud(request: PredictionRequest) -> Dict[str, Any]:
    try:
        prediction = await predictor.predict(request.dict())
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
