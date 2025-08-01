# MLOps Project: Obesity Risk Classification

## Project Overview
This project aims to develop a production-driven machine learning pipeline that classifies individuals into one of four obesity risk categories: Underweight, Normal weight, Overweight, and Obese. The goal is to simulate an end-to-end MLOps workflow from data acquisition to model evaluation and automation of code quality checks.

I chose the Kaggle Playground Series - Season 4, Episode 2 dataset for several reasons: (1) It contains real-world health and lifestyle attributes that influence obesity, making it suitable for feature engineering, model tuning, and fairness considerations; (2) the dataset is small enough for rapid experimentation yet rich enough for modeling tasks aligned with our learning goals.

This project utilizes Docker to ensure environment consistency by encapsulating dependencies and runtime configurations inside containers, effectively creating immutable infrastructure. This removes the "works on my machine" problem. And, Apache Airflow enables scalable orchestration of ML workflows, including retry mechanisms for flaky tasks and automatic logging. These tools together allow us to manage, test, and deploy machine learning pipelines in a reliable and reproducible way.

## How to Get the Data
1. Visit the competition page on Kaggle:
👉 https://www.kaggle.com/competitions/playground-series-s4e2/data
2. Download `train.csv` and place it inside the following directory:
```bash
data/raw/obesity_data.csv
```

## Setup Instructions
### Method 1: Run ML Pipeline Using Docker Setup
#### Step 1: Docker Installation
- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [Docker Engine for Linux](https://docs.docker.com/engine/install/)

After installation, verify Docker is working:
```bash
docker --version
```

#### Step 2: Clone the repository
Use the following command to download the project code:
```bash
git clone https://github.com/navibooy/e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8_obesity_risk_classification.git
```

#### Step 3: Build the Docker Image
Build the custom Docker image based on the `pipeline.Dockerfile` by using the following command:
```bash
docker build -f deploy/docker/pipeline.Dockerfile -t e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8-ml-pipeline .
```
#### Step 4: Run the Docker Container
This will execute the full pipeline using `src/run_pipeline.py` inside the container.
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/reports:/app/reports" \
e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8-ml-pipeline
```

### Method 2: Pipeline Orchestration via Airflow Containerized Setup
This method gives you full orchestration, scheduling and logging via Apache Airflow in a Dockerized environment.

#### Step 1: Install Docker & Docker Compose
Ensure Docker and Docker Compose are installed on your machine. Use these documentation for reference:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

To verify installation:
```bash
docker --version
docker compose version
```

#### Step 2: Clone the repository
Use the following command to download the project code:
```bash
git clone https://github.com/navibooy/e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8_obesity_risk_classification.git
```

#### Step 3: Start and Build the Airflow Services
This command initializes and builds all required services using Docker Compose:
```bash
docker compose -f deploy/docker/docker-compose.yaml up --build -d
```
This sets up the Airflow scheduler, triggerer, PostgreSQL, and Redis containers in detached mode.

#### Step 4: Run the Pipeline using Airflow UI
Open your browser and visit `http://localhost:8080/dags` to access Airflow web interface. Login using the default credentials:
- Username: `airflow`
- Password: `airflow`

In the UI, search for the DAG named `obesity_classification_dag` and trigger it manually. You can view logs, task status, and execution timelines here.

#### Step 5: Test the Pipeline using command line
You can also test individual tasks from the CLI. For example, to test the `preprocess_data` task:
```bash
docker compose -f deploy/docker/docker-compose.yaml exec airflow-scheduler airflow tasks test obesity_classification_dag preprocess_data 2025-08-01
```
Replace the date with your target execution date in `YYYY-MM-DD` format.

### Method 3: Local Setup
#### Step 1: Clone the repository
```bash
git clone https://github.com/navibooy/e33a2eeeb9e69ccce9e871fb11133de4856d640937543497fe75954637fa20c8_obesity_risk_classification.git
```

#### Step 2: Install UV (if not yet installed)
I used **UV** (a modern package and environment manager) for setting up a clean and reproducible virtual environment.
```bash
pip install uv
```
#### Step 3: Create a virtual environment with Python 3.10
```bash
uv venv --python=3.10
```
#### Step 4: Activate the environment
- Git Bash:
```bash
source .venv/Scripts/activate
```
- PowerShell:
```powershell
.venv\Scripts\activate.ps1
```
#### Step 5: Install dependencies from `pyproject.toml`
```bash
uv pip install -r pyproject.toml
```
#### Step 6: Verify installation (optional)
```bash
uv pip list
```
#### Step 7: Install and enable pre-commit hooks
```bash
pre-commit install
```
#### Step 8: Run the prediction and evaluate model
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
Throughout my journey in the Masters of Data Science at AIM, one question has persistently sparked my curiosity: *How are models actually deployed and made useful for end-users?* Until now, our model pipelines has primarily lived within Jupyter notebooks which is not ideal for production environments. This course has been particularly exciting because it addresses that curiosity by exposing us to the tools and workflows involved in deploying machine learning models effectively.

One of the biggest learning milestones for me was diving into Git and GitHub. While I've used version control before at a basic level, this was the first time I truly understood how essential it is for structuring production-ready machine learning projects. The emphasis on clean, trackable, and collaborative development using these tools was eye-opening and very practical for industry scenarios.

My passion for automation made me especially enthusiastic about learning how `.yaml` configuration files work, how to structure directories properly, and how to write modular, reusable code to support both data and model pipelines. These are foundational skills for creating scalable ML systems, and I found great satisfaction in putting them into practice.

With this I have also encoutered challenges in this project. Setting up the virtual environment using `uv` was new for me, and understanding how `pyproject.toml` works required additional research. I also ran into issues with pre-commit hooks—specifically, a loop where failed checks would revert my staged commits. This forced me to debug and tweak the `.pre-commit-config.yaml` file, teaching me how important it is to understand configuration and dependency management at a deeper level.

Another significant learning point was revisiting old, non-modular code I had written before. Refactoring it to be more atomic and maintainable was a valuable exercise that showed me how much better modular design is for testing, debugging, and scaling.

Looking ahead, I'm excited to continue exploring tools like Apache Airflow, Docker, and MLflow, which are powerful enablers of automation, reproducibility, and collaborative workflows in real-world ML operations.
