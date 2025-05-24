import logging
import os
from datetime import timedelta
from typing import NoReturn

import torch

from data_loader import get_data_bundle, log_data_bundle
from seed_initializer import seed_all, create_torch_generator, seed_worker

from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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

def train_model_with_mlflow():
    """Обучение модели с логированием в MLFlow"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all()

    b_folder = "~/medical-image-processing/xray-classifier/"
    clean_train_dir = b_folder + "datasets/chest_xray_clean/train"
    test_dir = b_folder + "datasets/chest_xray_clean/test"

    data = get_data_bundle(
        clean_train_dir=clean_train_dir,
        test_dir=test_dir,
        generator=create_torch_generator(),
        seed_worker_fn=seed_worker,
        image_size=(224, 224),
        batch_size=32,
        num_workers=0,  # Установите на 0 для Airflow
    )
    log_data_bundle(data)


def evaluate_and_register_model():
    """Оценка модели и регистрация в MLFlow если качество улучшилось"""
    pass


fetch_data_task = BashOperator(
    task_id='fetch_data_from_dvc_remote',
    bash_command='bash \'/opt/airflow/dags/pull_data_from_dvc.sh\'',
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
