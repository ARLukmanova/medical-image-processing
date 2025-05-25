import logging
import os
import mlflow
from datetime import timedelta
from typing import NoReturn

import torch
import torch.optim as optim
import torch.nn as nn

from data_loader import get_data_bundle, log_data_bundle
from seed_initializer import seed_all, create_torch_generator, seed_worker
from hybrid_cnn_transformer import HybridCNNTransformer
from train_model import train_model_mlflow

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
    print(device)
    seed_all()


    public_server_ip = os.environ.get('PUBLIC_SERVER_IP')
    ml_flow_public_port = os.environ.get('MLFLOW_PUBLIC_PORT')
    ml_flow_uri = f"http://{public_server_ip}:{ml_flow_public_port}/"
    print(f"MLFlow URI: {ml_flow_uri}")

    mlflow.set_tracking_uri(ml_flow_uri)
    mlflow.set_experiment("Hybrid-Training-CPU")

    with mlflow.start_run(log_system_metrics=True):

        # dataloaders
        proj_folder = "/home/airflow/medical-image-processing/xray-classifier/"
        clean_train_dir = proj_folder + "datasets/chest_xray_clean/train"
        test_dir = proj_folder + "datasets/chest_xray_clean/test"

        data_bundle = get_data_bundle(
            clean_train_dir=clean_train_dir,
            test_dir=test_dir,
            generator=create_torch_generator(),
            seed_worker_fn=seed_worker,
            image_size=(224, 224),
            batch_size=32,
            num_workers=0,  # Установите на 0 для Airflow
        )
        log_data_bundle(data_bundle)

        # Обучение модели

        model = HybridCNNTransformer(num_classes=4).to(device)
        pretrained_model_path = proj_folder + 'pretrained/hybrid_oct.pth'
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.change_num_classes(2, device)

        optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(weight=data_bundle.classes_info.weights.to(device))
        model, train_loss, train_acc, val_loss, val_acc = train_model_mlflow(
            model, data_bundle.loaders.train, data_bundle.loaders.val, criterion, optimizer, num_epochs=10,
            model_name='best_hybrid_oct'
        )


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
