import os
from pathlib import Path

class Config:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODEL_PATH = ARTIFACTS_DIR / "models" / "xgb_model_graph.bin"
    ENCODERS_DIR = ARTIFACTS_DIR / "label_encoders"
    
    # Model configuration
    FEATURE_COLUMNS = [
        'Avg_Mdcr_Alowd_Amt', 'Avg_Mdcr_Pymt_Amt', 'Avg_Mdcr_Stdzd_Amt',
        'Avg_Sbmtd_Chrg', 'Tot_Bene_Day_Srvcs', 'Tot_Benes', 'Tot_Srvcs',
        'Rndrng_Prvdr_Gndr', 'HCPCS_Drug_Ind', 'Place_Of_Srvc',
        'Rndrng_Prvdr_Type_encoded', 'HCPCS_Cd_encoded',
        'HCPCS_Degree_Centrality', 'HCPCS_Closeness_Centrality', 'HCPCS_PageRank',
        'Provider Type_Closeness_Centrality', 'Provider Type_PageRank'
    ]
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Redis configuration for caching
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))