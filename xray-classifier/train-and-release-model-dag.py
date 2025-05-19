import logging
import os
from datetime import timedelta
from typing import NoReturn

from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

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


def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)
    _LOG.info(os.getenv('MLFLOW_TRACKING_URI', 'NO_URI'))



def init() -> NoReturn:
    print('Hello, DAG!')
    bucket_name = Variable.get("S3_BUCKET")
    _LOG.info(f"Hello, DAG! Bucket name is {bucket_name}")
    configure_mlflow()


def use_s3(os=os) -> NoReturn:
    _LOG.info(os.getenv('MLFLOW_TRACKING_URI', 'NO_URI'))
    s3_hook = S3Hook("s3_connection")
    bucket_name = Variable.get("S3_BUCKET")
    import os
    _LOG.info(os.getcwdb())


task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)
task_use_s3 = PythonOperator(task_id='use_s3', python_callable=use_s3, dag=dag)

task_init >> task_use_s3
