from datetime import datetime

from airflow.operators.python import PythonOperator

from airflow import DAG
from src.data_preprocessing import read_and_preprocess_data
from src.evaluation import evaluate_model
from src.feature_engineering import split_and_engineer_features
from src.model_training import train_model

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 7, 30),
    "retries": 1,
}

with DAG(
    dag_id="obesity_classification_dag",
    default_args=default_args,
    description="Obesity classification ML pipeline DAG",
    schedule_interval=None,
    catchup=False,
    tags=["obesity", "ml", "training"],
) as dag:
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=lambda: read_and_preprocess_data("data/raw/obesity.csv"),
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=lambda: split_and_engineer_features(
            read_and_preprocess_data("data/raw/obesity.csv")
        ),
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=lambda: train_model(
            *split_and_engineer_features(
                read_and_preprocess_data("data/raw/obesity.csv")
            )
        ),
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=lambda: evaluate_model(
            model_path="models/model.pkl",
            X_test=split_and_engineer_features(
                read_and_preprocess_data("data/raw/obesity.csv")
            )[1],
            y_test=split_and_engineer_features(
                read_and_preprocess_data("data/raw/obesity.csv")
            )[3],
        ),
    )

    (
        preprocess_task
        >> feature_engineering_task
        >> train_model_task
        >> evaluate_model_task
    )
