# MLOps Project: Obesity Risk Classification

## Table of Contents
- [Project Overview](#project-overview)
- [How to Get the Data](#how-to-get-the-data)
- [Setup Instructions](#setup-instructions)
- [MLFlow Integration](#mlflow-integration)
- [Model Drift Detection](#model-drift-detection)
- [Folder Structure](#folder-structure)
- [Testing Instructions](#testing-instructions)
- [Reflection - HW3](#reflection---hw3)

## Project Overview
This project aims to develop a production-driven machine learning pipeline that classifies individuals into one of four obesity risk categories: Underweight, Normal weight, Overweight, and Obese. The goal is to simulate an end-to-end MLOps workflow from data acquisition to model evaluation and automation of code quality checks.

I chose the Kaggle Playground Series - Season 4, Episode 2 dataset for several reasons: (1) It contains real-world health and lifestyle attributes that influence obesity, making it suitable for feature engineering, model tuning, and fairness considerations; (2) the dataset is small enough for rapid experimentation yet rich enough for modeling tasks aligned with our learning goals.

This project utilizes Docker to ensure environment consistency by encapsulating dependencies and runtime configurations inside containers, effectively creating immutable infrastructure. This removes the "works on my machine" problem. And, Apache Airflow enables scalable orchestration of ML workflows, including retry mechanisms for flaky tasks and automatic logging. These tools together allow us to manage, test, and deploy machine learning pipelines in a reliable and reproducible way.

## How to Get the Data
1. Visit the competition page on Kaggle:
👉 https://www.kaggle.com/competitions/playground-series-s4e2/data or,
2. Simply run the following command below:
```bash
python src/data_ingestion.py
```

## Setup Instructions

### Method 1: Run ML Pipeline Using Docker Setup
### Prerequisites
- **Docker**: Version 20.10+ with Docker Compose support
- **Python**: Version 3.10+ (for local development)
- **Git**: For repository cloning and branch management
- **Minimum 8GB RAM**: Required for MLflow and Airflow services

### Step 1: Repository Setup
Clone the repository and checkout the hw3-mlflow-drift branch:
```bash
git clone https://github.com/navibooy/e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8_obesity_risk_classification.git

cd e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8_obesity_risk_classification

git checkout hw3-mlflow-drift
```

### Step 2: Directory Structure Setup
Create required directories for MLflow and pipeline outputs:
```bash
mkdir -p mlflow/runs mlflow/artifacts reports data/splits airflow_results
```

### Step 3: Docker Compose Startup
Start all services including MLflow, Airflow, PostgreSQL, and Redis:
```bash
docker-compose -f deploy/docker/docker-compose.yaml up -d
```

### Step 4: UI Access
- **MLflow UI**: http://localhost:5000 - View experiments, runs, and model registry
- **Airflow UI**: http://localhost:8080 - Monitor DAG execution and task logs
  - Username: `airflow`
  - Password: `airflow`

### Step 5: Trigger DAG in Airflow UI
Navigate to Airflow UI and go to Dags, click on ml_pipeline_dag, and proceed with Trigger DAG.

## MLFlow Integration

### Experiment Tracking Setup
The project uses MLflow for comprehensive experiment tracking and model management:

- **Experiment Name**: `obesity_risk_classification`
- **Tracking URI**:
  - Docker environment: `http://mlflow:5000` (container-to-container communication)
  - Local development: `http://localhost:5000` (host access)
- **Backend Store**: PostgreSQL database for metadata storage
- **Artifact Store**: Local filesystem at `mlflow/artifacts/`

### Model Registration and Logging
Each model training run automatically logs:

**Hyperparameters (3 required parameters):**
- `n_estimators`: Number of boosting rounds for XGBoost
- `max_depth`: Maximum tree depth for complexity control
- `learning_rate`: Step size shrinkage for gradient boosting

**Metrics (2 core metrics with variants):**
- `accuracy`: Classification accuracy on test set
- `f1_score`: F1-score for balanced evaluation
- `accuracy_original`: Performance on original test data
- `accuracy_drifted`: Performance on drift-simulated data
- `f1_score_original`: F1-score on original test data
- `f1_score_drifted`: F1-score on drift-simulated data

**Custom PyFunc Wrapper:**
The project implements a custom MLflow PyFunc wrapper that:
- Handles preprocessing pipeline integration
- Manages feature name consistency
- Provides standardized prediction interface
- Enables model versioning and deployment

**Artifacts and Metadata:**
- Model binary files (`.pkl` format)
- Feature names mapping (`feature_names.txt`)
- Preprocessing pipeline objects
- Training configuration files

### MLflow UI Access and Navigation

**Accessing the UI:**
Navigate to http://localhost:5000 to access the MLflow web interface.

**Key Navigation Areas:**
- **Experiments**: View all runs for the `obesity_risk_classification` experiment
- **Models**: Access the model registry and version management
- **Compare Runs**: Side-by-side comparison of hyperparameters and metrics
- **Artifact Browser**: Download and inspect saved model artifacts

**What to Look For:**
- **Parameters Tab**: View the 3 logged hyperparameters for each run
- **Metrics Tab**: Track accuracy and F1-score variations across runs
- **Artifacts Tab**: Access model files, feature mappings, and metadata
- **Tags**: Run metadata including environment detection and drift status

### Method 2: Local Development Setup (Alternative)
For local development without Docker:

```bash
# Install UV package manager
pip install uv

# Create virtual environment
uv venv --python=3.10

# Activate environment (Windows Git Bash)
source .venv/Scripts/activate

# Install dependencies
uv pip install -r pyproject.toml

# Install pre-commit hooks
pre-commit install


# Ensure that the port is running
mlflow server --host 127.0.0.1 --port 5000

# Run pipeline locally
python src/run_pipeline.py
```

## Model Drift Detection

The project implements a comprehensive drift detection system using the Evidently library with the DataDriftPreset to monitor model performance degradation over time. The system employs sophisticated statistical methods to detect when the distribution of incoming data significantly deviates from the training distribution. For numerical features, drift simulation is performed by applying a 1.2x multiplier to feature means and introducing Gaussian noise with a standard deviation of 0.1 times the original feature's standard deviation, mimicking realistic data distribution shifts. Categorical features undergo drift simulation through random value flipping, where 10-15% of categorical values are randomly reassigned to different valid categories within each feature's domain. The system calculates comprehensive drift scores using statistical tests and sets configurable thresholds (default 0.5) for automated decision-making. When drift scores exceed the defined threshold, the system automatically triggers model retraining workflows through Airflow DAG execution, ensuring continuous model accuracy. This approach provides real-world applications including early warning systems for model degradation, automated quality assurance for production ML systems, and proactive maintenance scheduling that prevents performance deterioration before it impacts business outcomes.

# Folder Structure
```bash
.
├── data/
│   ├── raw/                     # Original dataset
│   ├── processed/               # Cleaned and feature-engineered dataset
│   └── splits/                  # Train/test data splits for original and drifted datasets
├── notebooks/
│   └── obesity_eda.ipynb        # Contains exploratory data analysis
├── models/                      # Saved XGBoost model and MLflow integration artifacts
├── reports/                     # Evaluation reports, confusion matrices, and drift analysis outputs
├── mlflow/                      # MLflow experiment tracking and artifact storage
│   ├── runs/                    # Individual experiment run metadata
│   └── artifacts/               # Model artifacts and feature mappings
├── airflow_results/             # Inter-task communication and pipeline outputs
├── config/                      # Configuration management and environment detection
├── src/
│   ├── data_preprocessing.py    # Cleans and transforms raw data
│   ├── feature_engineering.py   # Splits, maps labels, resamples
│   ├── model_training.py        # Trains pipeline with MLflow logging
│   ├── evaluation.py            # Evaluates model performance on original and drifted data
│   ├── drift_detection.py       # Drift simulation and detection using Evidently
│   └── run_pipeline.py          # Executes the full pipeline
├── tests/                       # pytest functions for unit testing
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   ├── test_evaluation.py
├── deploy/                      # Deployment utilities for running the pipeline in production
│   ├── airflow                  # DAG scripts and Airflow-related files
│   ├── docker                   # Dockerfiles and docker-compose setup for both pipeline and orchestration
├── .pre-commit-config.yaml      # Code quality hooks
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Dependency backup (optional)
├── README.md                    # Project overview and setup guide
```

## HW3 Directory Justification

The enhanced folder structure for HW3 introduces several critical directories that support the complete ML lifecycle from development to production monitoring. The `mlflow/` directory with its `runs/` and `artifacts/` subdirectories provides centralized experiment tracking and model versioning, enabling reproducible research and systematic hyperparameter optimization. The `reports/` directory has been expanded to accommodate drift detection outputs and comparative analysis between original and drifted model performance, facilitating comprehensive model monitoring. The `src/` structure now includes `drift_detection.py` for automated data drift simulation and detection, implementing real-world monitoring capabilities that are essential for production ML systems. The `models/` directory integrates seamlessly with MLflow for both local model storage and remote artifact management. The `airflow_results/` directory enables robust inter-task communication and provides persistent storage for pipeline outputs, supporting complex workflow orchestration. The `config/` directory centralizes environment detection and configuration management, ensuring smooth transitions between development and production environments. This separation of concerns between data management, model artifacts, monitoring outputs, and metadata creates a scalable architecture that supports collaborative development, automated testing, and production deployment while maintaining clear data lineage and experiment reproducibility essential for enterprise ML operations.

## Testing Instructions

### 1. Standalone Pipeline Testing
Test the complete ML pipeline outside of Airflow:
```bash
# Ensure that the port is running
mlflow server --host 127.0.0.1 --port 5000

# Run complete pipeline (expect drift error message for demonstration)
python src/run_pipeline.py
```

**Expected Output**: The pipeline will execute successfully but display a drift detection error message, demonstrating the drift monit
oring system.

**Success Indicators**:
- All preprocessing steps complete without errors
- Model training completes and saves artifacts
- MLflow logging shows successful run with parameters and metrics
- Drift detection runs and reports drift status

### 2. Airflow DAG Testing

**DAG Syntax Verification**:
```bash
# Test DAG structure
airflow dags test ml_pipeline_dag 2025-08-20
```

### 3. MLflow Verification

**API Endpoint Testing**:
```bash
# Test MLflow API health
curl -f http://localhost:5000/health
```

**UI Verification Checklist**:
- [ ] Access MLflow UI at http://localhost:5000
- [ ] Verify `obesity_risk_classification` experiment exists
- [ ] Check that runs show logged parameters: `n_estimators`, `max_depth`, `learning_rate`
- [ ] Verify metrics are logged: `accuracy`, `f1_score`, and their variants
- [ ] Confirm artifacts are stored: model files and feature mappings

### 4. Validation Checklist

Complete HW3 requirements verification:

- [✅] **Docker Setup**: All containers start successfully
- [✅] **MLflow Integration**: Experiment tracking works with 3 parameters and 2 metrics
- [✅] **Drift Detection**: System simulates and detects drift correctly
- [✅] **Airflow DAG**: All tasks execute without errors
- [✅] **UI Access**: Both MLflow (port 5000) and Airflow (port 8080) UIs accessible
- [✅] **Model Registration**: Models are logged with custom PyFunc wrapper
- [✅] **Artifact Storage**: Model artifacts and metadata stored correctly
- [✅] **Environment Detection**: System correctly identifies Docker vs local environment
- [✅] **Inter-task Communication**: XCom and file-based communication working
- [✅] **Error Handling**: Graceful handling of MLflow connection issues
- [✅] **Drift Simulation**: Numerical and categorical drift simulation functional
- [✅] **Automated Retraining**: Drift detection triggers appropriate responses
- [✅] **Data Lineage**: Clear tracking from raw data through model artifacts

## Reflection - HW3

### Challenges Encountered

**MLflow Container Communication Issues:**
The most significant challenge involved establishing proper communication between Airflow tasks and the MLflow tracking server. Initially, tasks failed when attempting to connect to MLflow using `localhost:5000` from within Docker containers. The issue stemmed from container networking where `localhost` refers to the container's internal network rather than the host machine. This required implementing smart environment detection to use `mlflow:5000` for container-to-container communication versus `localhost:5000` for local development.

**Parameter Logging Failures and Silent Errors:**
MLflow parameter logging initially failed silently, making debugging extremely challenging. The system appeared to function correctly but parameters weren't being logged to experiments. Investigation revealed that MLflow run context wasn't properly established before logging calls, and error handling wasn't comprehensive enough to surface these failures. Additionally, parameter validation wasn't occurring, leading to type mismatches and formatting issues that caused silent failures.

**Docker Environment Detection Complexity:**
Differentiating between Docker and local execution environments proved more complex than anticipated. Simple environment variable checks were insufficient because multiple execution contexts existed: local development, Docker Compose orchestration, and Airflow container execution. Each context required different configuration approaches for MLflow connectivity, file paths, and service discovery.

**Airflow XCom Authentication Issues:**
Airflow version 3.0.3 introduced stricter authentication requirements for XCom operations, causing inter-task communication failures. Tasks couldn't access XCom data due to permission restrictions and authentication token management issues, breaking the pipeline's data flow between preprocessing, training, and evaluation stages.

### Solutions Implemented

**Smart Environment Detection System:**
Developed a comprehensive environment detection system in `config_loader.py` that analyzes multiple indicators including environment variables (`DOCKER_ENV`), hostname patterns, filesystem characteristics, and network connectivity tests. This system dynamically configures MLflow tracking URIs and communication protocols based on the detected environment, ensuring seamless operation across development and production contexts.

**MLflow Run Context Verification and Error Handling:**
Implemented robust MLflow run context management with explicit verification before any logging operations. Added comprehensive error handling that captures and logs MLflow connection failures, parameter type mismatches, and silent errors. Introduced run context validation that ensures active runs exist before attempting parameter or metric logging, with automatic run creation when necessary.

**Dual Communication Strategy:**
Architected a dual communication strategy using both XCom and file-based fallbacks for inter-task data sharing. When XCom authentication fails, the system automatically falls back to JSON file storage in the `airflow_results/` directory, ensuring pipeline continuity regardless of XCom authentication status. This approach provides resilience against Airflow version-specific authentication changes.

**Comprehensive Logging and Monitoring:**
Enhanced the entire pipeline with structured logging that provides visibility into MLflow connectivity status, parameter logging success/failure, drift detection results, and inter-task communication methods. Added health checks and diagnostic commands that enable proactive identification of configuration issues before they impact pipeline execution.

### Technical Improvements and Architecture Decisions

The HW3 implementation introduced several architectural improvements including centralized configuration management, environment-aware service discovery, and fault-tolerant inter-service communication. The decision to implement custom PyFunc wrappers for MLflow model registration enabled standardized deployment interfaces while maintaining preprocessing pipeline integration. The choice to use Evidently for drift detection provided statistical rigor while remaining computationally efficient for production monitoring. These decisions collectively created a robust, scalable ML pipeline capable of handling real-world production challenges including service failures, network partitions, and version compatibility issues. The experience reinforced the importance of comprehensive error handling, environment abstraction, and graceful degradation in distributed ML systems.
