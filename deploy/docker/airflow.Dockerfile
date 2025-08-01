# Start from the official Airflow image
FROM apache/airflow:3.0.3-python3.10

# Set working directory
WORKDIR /opt/airflow

# Install system packages required by data science tools
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install uv (modern dependency manager)
RUN pip install --no-cache-dir "uv>=0.1.24"
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml

COPY src/ /opt/airflow/src/
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/src"
