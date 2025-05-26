from datetime import timedelta

from airflow.models import DAG, Variable, TaskInstance
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago



PROJECT_PATH = "/home/airflow/medical-image-processing/xray-classifier/"


dag = DAG(
    dag_id="train-and-release-model",
    # schedule_interval='0 1 * * *',
    start_date=days_ago(2),
    catchup=False,
    tags=[],
    default_args={
        'owner': 'Alina Lukmanova',
        'email': 'arlukmanova@edu.hse.ru',
        'email_on_failure': True,
        'email_on_retry': False,
        'retry': 1,
        'retry-delay': timedelta(minutes=1)

    }
)

def train_model_with_logging():
    from data_loader import get_data_bundle
    from parameters import MODEL_NAME, USE_GPU
    from seed_initializer import seed_all
    from train_model import train_model
    from track_model import save_model_as_onnx_file
    import mlflow
    import torch
    from track_model import init_mlflow

    """Обучение модели с логированием в MLFlow"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if USE_GPU:
        assert device == torch.device('cuda')

    seed_all()

    init_mlflow()
    active_run = mlflow.start_run(log_system_metrics=True)

    data_bundle = get_data_bundle(PROJECT_PATH, num_workers=0)  # 0 для Airflow

    model, train_loss, train_acc, val_loss, val_acc = train_model(
        proj_path=PROJECT_PATH,
        data_bundle=data_bundle,
        device=device,
        model_name=MODEL_NAME,
        dry_run=not USE_GPU,
    )

    run_id = active_run.info.run_id
    mlflow.end_run()
    save_model_as_onnx_file(model, run_id)

    return run_id


def evaluate_and_register_model(ti: TaskInstance):
    import mlflow
    from track_model import publish_onnx_model_to_registry
    from track_model import init_mlflow
    run_id = ti.xcom_pull(task_ids=train_model_task.task_id)

    init_mlflow()
    mlflow.start_run(run_id)
    publish_onnx_model_to_registry(run_id, make_current=True)
    mlflow.end_run()


fetch_data_task = BashOperator(
    task_id='fetch_data_from_dvc_remote',
    bash_command='bash \'/opt/airflow/dags/pull_data_from_dvc.sh\'',
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model_with_logging',
    python_callable=train_model_with_logging,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_and_register_model',
    python_callable=evaluate_and_register_model,
    dag=dag,
)

fetch_data_task >> train_model_task >> evaluate_model_task
