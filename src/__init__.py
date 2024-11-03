from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent

# Define paths to various resources
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ENCODERS_DIR = ARTIFACTS_DIR / "label_encoders"
GRAPHS_DIR = ARTIFACTS_DIR / "graphs"

# Create directories if they don't exist
for directory in [ARTIFACTS_DIR, MODELS_DIR, ENCODERS_DIR, GRAPHS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
