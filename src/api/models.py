from pydantic import BaseModel, Field, validator
from typing import List, Optional


class ClaimInput(BaseModel):
    Rndrng_NPI: str = Field(..., description="Provider NPI")
    HCPCS_Cd: str = Field(..., description="HCPCS Code")
    Rndrng_Prvdr_Type: str = Field(..., description="Provider Type")
    Rndrng_Prvdr_Gndr: str = Field(..., enum=["M", "F"], description="Provider Gender")
    HCPCS_Drug_Ind: str = Field(
        ..., enum=["Y", "N"], description="HCPCS Drug Indicator"
    )
    Place_Of_Srvc: str = Field(..., enum=["F", "O"], description="Place of Service")
    Avg_Mdcr_Alowd_Amt: float = Field(
        ..., ge=0, description="Average Medicare Allowed Amount"
    )
    Avg_Mdcr_Pymt_Amt: float = Field(
        ..., ge=0, description="Average Medicare Payment Amount"
    )
    Avg_Mdcr_Stdzd_Amt: float = Field(
        ..., ge=0, description="Average Medicare Standardized Amount"
    )
    Avg_Sbmtd_Chrg: float = Field(..., ge=0, description="Average Submitted Charge")
    Tot_Bene_Day_Srvcs: int = Field(
        ..., ge=0, description="Total Beneficiary Day Services"
    )
    Tot_Benes: int = Field(..., ge=0, description="Total Beneficiaries")
    Tot_Srvcs: int = Field(..., ge=0, description="Total Services")

    @validator("Rndrng_NPI")
    def validate_npi(cls, v):
        if not v.isdigit() or len(v) != 10:
            raise ValueError("NPI must be a 10-digit number")
        return v


class BatchClaimInput(BaseModel):
    claims: List[ClaimInput]


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    feature_importance: Optional[dict]


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    summary: dict


class ErrorResponse(BaseModel):
    error: str
    details: Optional[dict] = None
