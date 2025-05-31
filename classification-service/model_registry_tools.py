import mlflow
import os
import shutil
from logger import logger
from parameters import MODEL_PATH, MODEL_NAME, LATEST_MODEL_VERSION_ALIAS, MODEL_DIR, MLFLOW_PORT, MLFLOW_IP, \
    MODEL_VERSION_FILENAME


def ensure_model_file_exists(force_model_update=False):
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Файл модели {MODEL_PATH} не найден. Загружаем из MLflow...")
        _download_model()
    elif force_model_update and (_get_model_version() != _get_registry_latest_model_version()):
        download_new_model_version()


def download_new_model_version():
    logger.info(f'В репозитории MLflow есть новая версия модели {MODEL_NAME}. Обновляем...')
    shutil.rmtree(MODEL_DIR)
    _download_model()


def _download_model():
    try:
        model_uri = f"models:/{MODEL_NAME}@{LATEST_MODEL_VERSION_ALIAS}"
        model_version = _get_registry_latest_model_version()
        os.makedirs(MODEL_DIR, exist_ok=True)
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=MODEL_DIR)
        _save_model_version(model_version)
        logger.info(f"Модель {MODEL_NAME} v{model_version} успешно скачана в {MODEL_DIR}")
    except Exception as e:
        logger.error(f"Ошибка при скачивании модели из MLflow: {e}")
        raise RuntimeError(f"Не удалось скачать модель из MLflow: {e}")
    return model_version


def init_mlflow():
    ml_flow_uri = f"http://{MLFLOW_IP}:{MLFLOW_PORT}/"
    mlflow.set_tracking_uri(ml_flow_uri)
    logger.info(f"MLFlow URI: {ml_flow_uri}")


def _save_model_version(model_version):
    with open(os.path.join(MODEL_DIR, MODEL_VERSION_FILENAME), "w", encoding="utf-8") as f:
        f.truncate(0)
        f.write(str(model_version))


def _get_model_version():
    with open(os.path.join(MODEL_DIR, MODEL_VERSION_FILENAME), "r", encoding="utf-8") as f:
        version = f.read().strip()
    logger.info(f"Текущая загруженная из репозитория версия модели {MODEL_NAME}: {version}")
    return version


def _get_registry_latest_model_version():
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, LATEST_MODEL_VERSION_ALIAS).version
    logger.info(f"Последняя версия модели {MODEL_NAME} в репозитории MLflow: {model_version}")
    return model_version
