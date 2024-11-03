# Healthcare Fraud Detection

This project aims to detect healthcare fraud using machine learning.

## Demo

Frontend: https://bpjs-healthkathon.streamlit.app/

## Installation

### Using Docker

```bash
docker compose up --build
```

### Using pip

```bash
pip install -r requirements.txt
```

```bash
streamlit run src/ui/app.py
```

```bash
uvicorn src.api.main:app --reload
```

## Project Structure

The project is structured as follows:

- `src/`: Source code for the project.

  - `api/`: API endpoints for the project.
  - `config.py`: Configuration for the project.
  - `data/`: Data preprocessing and utility functions.
  - `models/`: Machine learning models and predictor class.
  - `ui/`: Streamlit app for the project.

- `tests/`: Unit tests for the project (not yet implemented).
- `artifacts/`: Directory for storing artifacts like models, label encoders, and graphs.
- `notbook/`: Jupyter notebooks for exploratory data analysis and model training.
- `docker-compose.yml`: Docker Compose file for running the project.
- `Dockerfile`: Dockerfile for building a Docker image for the project.
- `requirements.txt`: List of dependencies for the project.
