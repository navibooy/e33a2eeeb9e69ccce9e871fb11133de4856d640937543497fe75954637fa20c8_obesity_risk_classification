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
│   └── processed/               # Cleaned and feature-engineered dataset
├── notebooks/
│   └── obesity_eda.ipynb        # Contains exploratory data analysis
├── models/                      # Saved XGBoost model
├── reports/                     # Evaluation reports and confusion matrix
├── src/
│   ├── data_preprocessing.py    # Cleans and transforms raw data
│   ├── feature_engineering.py   # Splits, maps labels, resamples
│   ├── model_training.py        # Trains pipeline and saves model
│   ├── evaluation.py            # Evaluates model performance
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

The modular folder structure reflects each step in the ML lifecycle (from data to model to reporting), making the project maintainable, reproducible, and aligned with MLOps practices.

# Docker Integration
This project uses two Dockerfiles and a centralized `docker-compose.yaml` file to manage reproducible environments for both ML pipeline execution and workflow orchestration.

`pipeline.Dockerfile`: Defines a minimal image for running `src/run_pipeline.py` outside of Airflow. This allows testing the ML logic independently using bind-mounted volumes for data, models, and reports.

`airflow.Dockerfile`: Extends the official `apache/airflow` image and installs required Python dependencies via `uv`. It's used in `docker-compose.yaml` to launch the Airflow services.

`docker-compose.yaml`: Coordinates multiple containers including Airflow webserver, scheduler, triggerer, PostgreSQL (metadata DB), and Redis (trigger event queue). It also mounts shared volumes (e.g., `data/`, `models/`, `reports/`) into `/opt/airflow` so all tasks can access input/output consistently across services.

This modular Docker strategy enables:
- Environment consistency across dev/staging/production
- Reusable pipeline code
- Clear separation between orchestration logic and ML model logic

# Airflow DAG
The DAG (`obesity_classification_dag`) is defined in `deploy/airflow/ml_pipeline_dag.py` and manages the ML workflow as a sequence of Python-based tasks using `PythonOperator`.

## DAG Structure:
1. `preprocess_data` – Cleans and prepares the raw CSV data.
2. `feature_engineering` – Splits, transforms, and optionally resamples the data.
3. `train_model` – Trains and saves the XGBoost model.
4. `evaluate_model` – Logs classification metrics and saves a confusion matrix.

These tasks are dependent on one another and executed in order using task chaining:
`preprocess_data >> feature_engineering >> train_model >> evaluate_model`

## Scheduling Rationale:
- The DAG is currently set to run manually (no fixed schedule) to facilitate iterative testing in development.
- catchup=False is used to avoid backfilling.
- Each task is idempotent and supports retry mechanisms by design.

# Pre-commit Configuration
I used pre-commit to enforce code quality and consistency automatically before every commit. This ensures that all contributors follow the same standards, and the codebase stays clean, readable, and production-ready.

Hooks used:

`ruff`: A fast Python linter that checks for stylistic errors, unused imports, and other issues. It is configured to auto-fix and return a non-zero exit if changes were made.

`pre-commit-hooks`: Includes general-purpose checks like fixing trailing whitespace, enforcing end-of-file newlines, validating YAML and TOML syntax, and detecting large files.

`yamllint`: Enforces consistent YAML formatting (e.g., indentation, line length) across .yaml and .yml files.

To enable and run pre-commit hooks:
```bash
pre-commit install
pre-commit run --all-files
```
Make sure you have all dependencies installed and your `.pre-commit-config.yaml` file is correctly defined at the project root.

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

# Reflection - HW2
Before learning new tools, I always ask myself what value does this bring to me? Through this homework, I came to understand that machine learning models do not always run consistently across different systems, since each machine has its own unique environment. And, Docker addresses this issue by providing a consistent and reproducible runtime environment, making model deployment and production workflows more reliable. Also, having prior experience with no-code workflow tools such as Power Automate, I was already familiar with the concept of task orchestration. This made me excited to explore Apache Airflow and apply its automation capabilities to a machine learning pipeline. One of the key challenges I encountered during this assignment was understanding the syntax and structure of Dockerfiles and `docker-compose.yaml`. I made a conscious effort to learn what each line of configuration meant, as it was essential to properly set up Docker in a way that would allow my machine learning pipeline to work seamlessly across different systems using containers. Also, handling Python path issues in Airflow tasks was a challenge. When I first ran the DAG, it failed to locate the modules in my `src/` directory due to missing or incorrect `PYTHONPATH` configuration inside the containerized Airflow environment. This was particularly confusing because the code ran perfectly outside of Docker. To resolve this, I explicitly set the `PYTHONPATH` environment variable in both the `Dockerfile` and `docker-compose.yaml`, ensuring it included the absolute path to `/opt/airflow/src`. I also verified that the `src/` directory was correctly copied or mounted into the container. This experience helped me better understand how environment isolation in Docker impacts module resolution and emphasized the importance of aligning paths between the host and container.
