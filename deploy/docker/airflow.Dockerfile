# Start from the official Apache Airflow image with Python 3.10
FROM apache/airflow:3.0.3-python3.10

# Set the working directory to the default Airflow home
WORKDIR /opt/airflow

# Switch to root user to install system dependencies
USER root

# Install essential system packages required by Python build tools and general development
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user for running Airflow safely
USER airflow

# Install uv (modern dependency manager)
RUN pip install --no-cache-dir "uv>=0.1.24"

# Copy dependency definitions into the container
COPY pyproject.toml ./

# Install Python packages as defined in pyproject.toml
RUN uv pip install -r pyproject.toml

# Copy custom source code for DAG tasks into the container
COPY src/ /opt/airflow/src/

# Add the src directory to PYTHONPATH so Airflow can import custom modules in DAGs
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/src"
