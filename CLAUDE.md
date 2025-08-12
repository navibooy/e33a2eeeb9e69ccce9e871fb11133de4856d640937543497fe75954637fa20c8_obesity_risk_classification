# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLOps project for obesity risk classification that uses Docker and Apache Airflow for pipeline orchestration. The project classifies individuals into four obesity risk categories: Underweight, Normal weight, Overweight, and Obese.

## Essential Commands

### Environment Setup
```bash
# Install UV package manager
pip install uv

# Create virtual environment with Python 3.10
uv venv --python=3.10

# Activate environment (Git Bash)
source .venv/Scripts/activate

# Install dependencies
uv pip install -r pyproject.toml

# Install pre-commit hooks
pre-commit install
```

### Development Commands
```bash
# Run the complete ML pipeline
python src/run_pipeline.py

# Run tests
pytest
pytest -v --tb=short

# Run pre-commit hooks
pre-commit run --all-files

# Lint code
ruff check --fix src/ tests/
```

### Docker Commands
```bash
# Build pipeline Docker image
docker build -f deploy/docker/pipeline.Dockerfile -t e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8-ml-pipeline .

# Run pipeline in Docker
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/reports:/app/reports" \
e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8-ml-pipeline

# Start Airflow services
docker compose -f deploy/docker/docker-compose.yaml up --build -d

# Test specific Airflow task
docker compose -f deploy/docker/docker-compose.yaml exec airflow-scheduler airflow tasks test obesity_classification_dag preprocess_data 2025-08-01
```

## Architecture

### Core ML Pipeline
The pipeline consists of 4 sequential steps orchestrated through both direct execution (`src/run_pipeline.py`) and Airflow DAG (`deploy/airflow/dags/ml_pipeline_dag.py`):

1. **Data Preprocessing** (`src/data_preprocessing.py`) - Cleans raw CSV data
2. **Feature Engineering** (`src/feature_engineering.py`) - Splits data, transforms features, handles imbalanced datasets
3. **Model Training** (`src/model_training.py`) - Trains XGBoost model
4. **Model Evaluation** (`src/evaluation.py`) - Generates metrics and confusion matrix

### Key Paths and Data Flow
- Raw data: `data/raw/obesity_data.csv` (download from Kaggle)
- Processed data: `data/processed/obesity_clean.csv`
- Train/test splits: `data/splits/` (X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl)
- Trained model: `models/model.pkl`
- Evaluation outputs: `reports/metrics.txt`, `reports/confusion_matrix.png`

### Docker Architecture
- **Pipeline Dockerfile** (`deploy/docker/pipeline.Dockerfile`) - Minimal image for running ML pipeline standalone
- **Airflow Dockerfile** (`deploy/docker/airflow.Dockerfile`) - Extends apache/airflow with project dependencies
- **Docker Compose** (`deploy/docker/docker-compose.yaml`) - Orchestrates Airflow services (scheduler, triggerer, PostgreSQL, Redis) with shared volumes

### Airflow Configuration
- DAG ID: `obesity_classification_dag`
- Manual trigger (no schedule)
- Default credentials: airflow/airflow
- Web UI: http://localhost:8080/dags
- All services share volumes for data, models, and reports at `/opt/airflow/`
- PYTHONPATH configured to `/opt/airflow` for module resolution

## Data Requirements

The project expects `train.csv` from Kaggle Playground Series - Season 4, Episode 2 to be placed at `data/raw/obesity_data.csv`.

## Testing

Tests are organized by module in `tests/` directory:
- `test_data_preprocessing.py`
- `test_feature_engineering.py`
- `test_model_training.py`
- `test_evaluation.py`

Test fixtures available in `tests/fixtures/sample_obesity_data.csv`.

## Code Quality

Pre-commit hooks enforce:
- Ruff linting with auto-fix
- Trailing whitespace removal
- End-of-file fixers
- YAML/TOML validation
- Large file detection
- YAML formatting via yamllint

## Dependencies

Main dependencies managed via `pyproject.toml`:
- Core ML: scikit-learn, xgboost, pandas, numpy
- Visualization: matplotlib
- Imbalanced data: imbalanced-learn
- Testing: pytest (dev dependency)
- Code quality: pre-commit, ruff (via pre-commit hooks)
