# MLOps Project: Obesity Risk Classification

## Project Overview
This project aims to develop a production-driven machine learning pipeline that classifies individuals into one of four obesity risk categories: Underweight, Normal weight, Overweight, and Obese. The goal is to simulate an end-to-end MLOps workflow from data acquisition to model evaluation and automation of code quality checks.

I chose the Kaggle Playground Series - Season 4, Episode 2 dataset for several reasons: (1) It contains real-world health and lifestyle attributes that influence obesity, making it suitable for feature engineering, model tuning, and fairness considerations; (2) The dataset is manageable in size (<100MB) and suitable for training with popular models like XGBoost, which aligns well with our coursework objectives.

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
We use pre-commit to enforce code quality and consistency automatically before every commit. This ensures that all contributors follow the same standards, and the codebase stays clean, readable, and production-ready.

`black`: Auto-formats Python code to a uniform style using PEP 8 rules. Helps prevent debates over spacing, quotes, etc.

`ruff`: A fast linter that detects issues like unused variables, undefined names, and stylistic errors. Replaces multiple linters like `flake8`, `pycodestyle`, and `pylint`.

`isort`: Automatically sorts and groups Python imports (standard, third-party, local) to improve readability and prevent merge conflicts.

`pre-commit-hooks`: Fixes trailing whitespace, YAML errors, large file commits, etc.

# Optional: Unit Testing with Pytest
This project uses `pytest` for unit testing to ensure code correctness and stability.

### Step 1: Install Pytest (if not yet installed)
```bash
uv add --dev pytest
```
### Step 2: Run Tests
Run all test:
```bash
pytest
```
OR add verbosity for clearer output:
```bash
pytest -v --tb=short
```

# Reflection
Throughout my journey in the Masters of Data Science at AIM, one question has persistently sparked my curiosity: *How are models actually deployed and made useful for end-users?* Until now, our model pipelines has primarily lived within Jupyter notebooks which is not ideal for production environments. This course has been particularly exciting because it addresses that curiosity by exposing us to the tools and workflows involved in deploying machine learning models effectively. One of the biggest learning milestones for me was diving into Git and GitHub. While I've used version control before at a basic level, this was the first time I truly understood how essential it is for structuring production-ready machine learning projects. The emphasis on clean, trackable, and collaborative development using these tools was eye-opening and very practical for industry scenarios.

My passion for automation made me especially enthusiastic about learning how `.yaml` configuration files work, how to structure directories properly, and how to write modular, reusable code to support both data and model pipelines. These are foundational skills for creating scalable ML systems, and I found great satisfaction in putting them into practice. With this I have also encoutered challenges in this project. Setting up the virtual environment using `uv` was new for me, and understanding how `pyproject.toml` works required additional research. I also ran into issues with pre-commit hooks—specifically, a loop where failed checks would revert my staged commits. This forced me to debug and tweak the `.pre-commit-config.yaml` file, teaching me how important it is to understand configuration and dependency management at a deeper level.

Another significant learning point was revisiting old, non-modular code I had written before. Refactoring it to be more atomic and maintainable was a valuable exercise that showed me how much better modular design is for testing, debugging, and scaling. Looking ahead, I'm excited to continue exploring tools like Apache Airflow, Docker, and MLflow, which are powerful enablers of automation, reproducibility, and collaborative workflows in real-world ML operations.
