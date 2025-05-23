import logging
import os
from datetime import timedelta
from typing import NoReturn

from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

dag = DAG(
    dag_id="train-and-release-model",
    schedule_interval='0 1 * * *',
    start_date=days_ago(2),
    catchup=False,
    tags=[],
    default_args={
        'owner': 'Alina Lukmanova',
        'email': 'arlukmanova@edu.hse.ru',
        'email_on_failure': True,
        'email_on_retry': False,
        'retry': 3,
        'retry-delay': timedelta(minutes=1)

    }
)


def fetch_data_with_dvc():
    """Получение данных с помощью DVC"""
    pass

def train_model_with_mlflow():
    """Обучение модели с логированием в MLFlow"""
    pass

def evaluate_and_register_model():
    """Оценка модели и регистрация в MLFlow если качество улучшилось"""
    pass

fetch_data_task = PythonOperator(
    task_id='fetch_data_from_dvc_remote',
    python_callable=fetch_data_with_dvc,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model_with_logging',
    python_callable=train_model_with_mlflow,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_and_register_model',
    python_callable=evaluate_and_register_model,
    dag=dag,
)

fetch_data_task >> train_model_task >> evaluate_model_task
