import os

import numpy as np
import onnx
import mlflow.onnx
import torch
from mlflow import MlflowClient

from parameters import MODEL_NAME, IMAGE_SIZE, LATEST_VERSION_MODEL_ALIAS, EXPERIMENT_NAME


def log_model_as_onnx(model, make_current=True):
    run_id = mlflow.active_run().info.run_id
    save_model_as_onnx_file(model, run_id)
    publish_onnx_model_to_registry(run_id, make_current)


def save_model_as_onnx_file(model, run_id):
    device = next(model.parameters()).device
    onnx_path = get_onnx_path(run_id)

    print(f'Сохраняем модель в формате ONNX: {onnx_path}')
    model.eval()
    torch.onnx.export(
        model,
        get_dummy_input().to(device),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    return onnx_path


def publish_onnx_model_to_registry(run_id, make_current):
    onnx_path = get_onnx_path(run_id)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Файл: {onnx_path} не был найден. Убедитесь, что модель была сохранена корректно, "
                                f"а также в том, что все шаги DAGа выполняются на одном и том же worker.")


    print(f'Загружаем модель в MLflow: {onnx_path}')
    log_result = mlflow.onnx.log_model(
        onnx_model=onnx.load(onnx_path),
        artifact_path="onnx_model",
        registered_model_name=MODEL_NAME,
        input_example={"input": get_dummy_input().cpu().numpy().astype(np.float32)},
        save_as_external_data=False
    )

    print(f'Удаляем временный файл с моделью: {onnx_path}')
    os.remove(onnx_path)
    if make_current:
        client = MlflowClient()
        run_id = log_result.run_id
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        version = None
        for v in versions:
            if v.run_id == run_id:
                version = v.version
                break
        if version:
            client.set_registered_model_alias(MODEL_NAME, LATEST_VERSION_MODEL_ALIAS, version)

    return log_result


def get_onnx_path(run_id):
    onnx_dir = "/tmp/medical-image-processing"
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = f"{onnx_dir}/hybrid_model_{run_id}.onnx"
    return onnx_path


def get_dummy_input():
    return torch.randn(1, 3, *IMAGE_SIZE)


def init_mlflow():
    public_server_ip = os.environ.get('PUBLIC_SERVER_IP')
    ml_flow_public_port = os.environ.get('MLFLOW_PUBLIC_PORT')
    ml_flow_uri = f"http://{public_server_ip}:{ml_flow_public_port}/"
    print(f"MLFlow URI: {ml_flow_uri}")
    mlflow.set_tracking_uri(ml_flow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
