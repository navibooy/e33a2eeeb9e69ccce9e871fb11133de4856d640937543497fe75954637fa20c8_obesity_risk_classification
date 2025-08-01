from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data_preprocessing import read_and_preprocess_data
from src.feature_engineering import split_and_engineer_features
from src.model_training import train_model
from src.evaluation import evaluate_model

RAW_PATH = "data/raw/obesity.csv"
PROCESSED_PATH = "data/processed/obesity_clean.csv"
SPLIT_DIR = "data/splits/"
MODEL_PATH = "models/model.pkl"
REPORT_PATH = "reports/metrics.txt"
PLOT_PATH = "reports/confusion_matrix.png"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 7, 31),
    "retries": 1,
}

with DAG(
    dag_id="obesity_classification_dag",
    default_args=default_args,
    description="Obesity classification ML pipeline DAG",
    schedule=None,
    catchup=False,
    tags=["obesity", "ml", "training"],
) as dag:

    def preprocess_task():
        read_and_preprocess_data(input_path=RAW_PATH, output_path=PROCESSED_PATH)

    def feature_engineering_task():
        split_and_engineer_features(
            input_path=PROCESSED_PATH,
            output_dir=SPLIT_DIR,
            resample=True
        )

    def train_model_task():
        train_model(
            X_train_path=f"{SPLIT_DIR}X_train.pkl",
            y_train_path=f"{SPLIT_DIR}y_train.pkl",
            model_path=MODEL_PATH
        )

    def evaluate_model_task():
        evaluate_model(
            model_path=MODEL_PATH,
            X_test_path=f"{SPLIT_DIR}X_test.pkl",
            y_test_path=f"{SPLIT_DIR}y_test.pkl",
            report_path=REPORT_PATH,
            plot_path=PLOT_PATH
        )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_task,
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_task,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_task,
    )

    preprocess >> feature_engineering >> train >> evaluate
