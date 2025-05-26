import os
from datetime import timedelta

from airflow.models import DAG, Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago


PROJECT_PATH = "/home/airflow/medical-image-processing/xray-classifier/"

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
    from data_loader import get_data_bundle
    from parameters import MODEL_NAME, EXPERIMENT_NAME
    from seed_initializer import seed_all
    from train_model import train_model
    from track_model import log_model_as_onnx
    import mlflow


    import torch
    """Обучение модели с логированием в MLFlow"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    seed_all()


    public_server_ip = os.environ.get('PUBLIC_SERVER_IP')
    ml_flow_public_port = os.environ.get('MLFLOW_PUBLIC_PORT')
    ml_flow_uri = f"http://{public_server_ip}:{ml_flow_public_port}/"
    print(f"MLFlow URI: {ml_flow_uri}")

    mlflow.set_tracking_uri(ml_flow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with (mlflow.start_run(log_system_metrics=True)):


        data_bundle = get_data_bundle(PROJECT_PATH, num_workers=0)  # 0 для Airflow

        model, train_loss, train_acc, val_loss, val_acc = train_model(
            proj_path=PROJECT_PATH,
            data_bundle=data_bundle,
            device=device,
            model_name=MODEL_NAME,
            dry_run=True,
        )

        log_model_as_onnx(model, make_current=True)


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
