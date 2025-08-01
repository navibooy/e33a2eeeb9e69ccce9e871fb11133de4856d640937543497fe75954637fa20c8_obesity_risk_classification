from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

# Function to run
def print_hello():
    print("✅ Hello from Airflow! Your DAG is working.")

# DAG definition
with DAG(
    dag_id="sample_python_dag",
    start_date=pendulum.today('UTC'),  # today’s date
    schedule=None,  # only trigger manually
    catchup=False,
    tags=["test"],
) as dag:

    hello_task = PythonOperator(
        task_id="say_hello",
        python_callable=print_hello,
    )
