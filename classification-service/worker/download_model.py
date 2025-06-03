from urllib.parse import urlparse
from tqdm import tqdm

import boto3
import mlflow
import os
import sys

MLFLOW_HOST = "188.72.77.22"
MLFLOW_PORT = "5050"
MODEL_FILENAME = "model.onnx"
MODEL_NAME = "xray-hybrid-classifier"
LATEST_MODEL_VERSION_ALIAS = "current"


def download_model():
    _init_mlflow()
    try:
        os.makedirs(model_dir, exist_ok=True)
        model_uri, model_version = _get_registry_latest_model_version_and_uri()
        print(f"Скачиваем модель {MODEL_NAME} v{model_version} из MLFlow Model Registry")

        _download_from_s3(model_uri)
        print(f"Модель {MODEL_NAME} v{model_version} успешно скачана в {model_dir}")

    except Exception as e:
        print(f"Ошибка при скачивании модели из MLFlow Model Registry: {e}")
        raise RuntimeError(f"Не удалось скачать модель из MLflow: {e}")
    return model_version


def _download_from_s3(model_uri):
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        region_name='ru-central1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    parsed = urlparse(model_uri)
    bucket = parsed.netloc
    s3_key = parsed.path.lstrip('/')
    head = s3.head_object(Bucket=bucket, Key=s3_key)
    file_size = head['ContentLength']
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    s3.download_file(bucket, s3_key, model_path, Callback=ProgressPercentage(s3_key, file_size))


def _init_mlflow():
    ml_flow_uri = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}/"
    mlflow.set_tracking_uri(ml_flow_uri)
    print(f"MLFlow URI: {ml_flow_uri}")


def _get_registry_latest_model_version_and_uri():
    client = mlflow.tracking.MlflowClient()
    model_version_info = client.get_model_version_by_alias(MODEL_NAME, LATEST_MODEL_VERSION_ALIAS)
    model_version = model_version_info.version
    model_uri = os.path.join(model_version_info.source, MODEL_FILENAME)
    print(f"Последняя версия модели {MODEL_NAME} - v.{model_version} "
          f"доступна в MLFlow Model Registry по URI: {model_uri}")
    return model_uri, model_version


class ProgressPercentage(object):
    def __init__(self, filename, size):
        self._filename = filename
        self._size = size
        self._seen_so_far = 0
        self._tqdm = tqdm(total=size, unit='B', unit_scale=True, desc=filename)

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._tqdm.update(bytes_amount)


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    aws_access_key_id = sys.argv[2] if len(sys.argv) > 2 else None
    aws_secret_access_key = sys.argv[3] if len(sys.argv) > 3 else None
    download_model()
