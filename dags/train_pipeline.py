from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

# DAG default args
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

# Define DAG
with DAG(
    "student_performance_training",
    default_args=default_args,
    description="Train and version student performance model with DVC + MLflow",
    schedule_interval=None,  # Trigger manually for now
    start_date=datetime(2025, 9, 1),
    catchup=False,
) as dag:

    # Step 1: Pull latest dataset from DVC
    dvc_pull = BashOperator(
        task_id="dvc_pull_data",
        bash_command="dvc pull data/raw/StudentPerformanceFactors.csv",
    )

    # Step 2: Run training script (produces models + logs in MLflow)
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python src/train.py",
    )

    # Step 3: Push new models back to DVC
    dvc_push = BashOperator(
        task_id="dvc_push_models",
        bash_command="dvc add models && git add models/.gitignore models/*.dvc && git commit -m 'Update models' && dvc push",
    )

    # Pipeline order
    dvc_pull >> train_model >> dvc_push
