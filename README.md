# MLOps Project: Obesity Risk Classification

## Project Overview
This project aims to develop a production-driven machine learning pipeline that classifies individuals into one of four obesity risk categories: Underweight, Normal weight, Overweight, and Obese. The goal is to simulate an end-to-end MLOps workflow from data acquisition to model evaluation and automation of code quality checks.

We chose the Kaggle Playground Series - Season 4, Episode 2 dataset for several reasons: (1) It contains real-world health and lifestyle attributes that influence obesity, making it suitable for feature engineering, model tuning, and fairness considerations; (2) The dataset is manageable in size (<100MB) and suitable for training with popular models like XGBoost, which aligns well with our coursework objectives.

## How to Get the Data
1. Visit the competition page on Kaggle:
👉 https://www.kaggle.com/competitions/playground-series-s4e2/data
2. Download `train.csv` and place it inside the following directory:
```bash
data/raw/obesity_data.csv
```

## Setup Instructions
I used **UV** (a modern package and environment manager) for setting up a clean and reproducible virtual environment.

### Step 1: Install UV (if not yet installed)
```bash
pip install uv
```
### Step 2: Create a virtual environment with Python 3.10
```bash
uv venv --python=3.10
```
### Step 3: Activate the environment
- Git Bash:
```bash
source .venv/Scripts/activate
```
- PowerShell:
```powershell
.venv\Scripts\activate.ps1
```
### Step 4: Install dependencies from `pyproject.toml`
```bash
uv pip install -r pyproject.toml
```
### Step 5: Verify installation (optional)
```bash
uv pip list
```
### Step 6: Install and enable pre-commit hooks
```bash
pre-commit install
```
### Step 7: Run the prediction and evaluate model
```bash
python src/run_pipeline.py
```

# Folder Structure
```bash
.
├── data/
│   ├── raw/                     # Original dataset
│   └── preprocessed/            # Cleaned and feature-engineered dataset
├── models/                      # Saved XGBoost model
├── reports/                     # Evaluation reports and confusion matrix
├── src/
│   ├── data_preprocessing.py    # Cleans and transforms raw data
│   ├── feature_engineering.py   # Splits, maps labels, resamples
│   ├── model_training.py        # Trains pipeline and saves model
│   ├── evaluation.py            # Evaluates model performance
│   └── run_pipeline.py          # Executes the full pipeline
├── .pre-commit-config.yaml      # Code quality hooks
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Dependency backup (optional)
├── README.md                    # Project overview and setup guide
```

The modular folder structure reflects each step in the ML lifecycle (from data to model to reporting), making the project maintainable, reproducible, and aligned with MLOps practices.

# Pre-commit Configuration
We use pre-commit to enforce clean, consistent, and professional code at every commit:
`black`: Enforces consistent code formatting

`ruff`: Fast linter for code issues (e.g., unused variables, import errors)

`isort`: Sorts and organizes imports logically

`pre-commit-hooks`: Fixes trailing whitespace, YAML errors, large file commits, etc.
