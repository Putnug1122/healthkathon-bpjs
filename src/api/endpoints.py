from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from src.models.predictor import FraudPredictor
from src.data.preprocessing import process_batch_claims
from .models import (
    ClaimInput,
    BatchClaimInput,
    PredictionResult,
    BatchPredictionResult,
    ErrorResponse,
)
import pandas as pd
from typing import List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Claims Fraud Detection API",
    description="API for detecting potential fraud in healthcare claims",
    version="1.0.0",
)

# Initialize the predictor
predictor = FraudPredictor()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.post("/predict/claim", response_model=PredictionResult)
async def predict_single_claim(claim: ClaimInput):
    """
    Predict fraud probability for a single claim
    """
    try:
        # Convert pydantic model to dict
        claim_dict = claim.dict()

        # Make prediction
        prediction, probability, features = predictor.predict(claim_dict)

        # Calculate feature importance
        feature_importance = dict(
            zip(features.columns, predictor.model.feature_importances_)
        )

        return PredictionResult(
            prediction="Fraud" if prediction == 1 else "No Fraud",
            confidence=probability[1] if prediction == 1 else probability[0],
            feature_importance=feature_importance,
        )

    except Exception as e:
        logger.error(f"Error processing claim: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Error processing claim", "message": str(e)},
        )


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch_claims(
    claims: BatchClaimInput, background_tasks: BackgroundTasks
):
    """
    Predict fraud probability for multiple claims
    """
    try:
        # Convert claims to DataFrame
        claims_df = pd.DataFrame([claim.dict() for claim in claims.claims])

        # Process claims
        results_df, error_rows = process_batch_claims(predictor, claims_df)

        # Log errors if any
        if error_rows:
            background_tasks.add_task(log_batch_errors, error_rows)

        # Prepare results
        prediction_results = [
            PredictionResult(
                prediction=row["Prediction"],
                confidence=row["Confidence"],
                feature_importance=None,  # Optional: could calculate for each claim if needed
            )
            for _, row in results_df.iterrows()
        ]

        # Calculate summary statistics
        summary = {
            "total_claims": len(results_df),
            "fraud_detected": len(results_df[results_df["Prediction"] == "Fraud"]),
            "processing_errors": len(error_rows),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return BatchPredictionResult(results=prediction_results, summary=summary)

    except Exception as e:
        logger.error(f"Error processing batch claims: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Error processing batch claims", "message": str(e)},
        )


async def log_batch_errors(error_rows: List[dict]):
    """Background task to log batch processing errors"""
    logger.error(f"Batch processing errors: {error_rows}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail["error"]), details={"message": exc.detail["message"]}
        ).dict(),
    )
