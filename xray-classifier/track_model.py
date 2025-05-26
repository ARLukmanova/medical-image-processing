import os

import numpy as np
import onnx
import mlflow.onnx
import torch
from mlflow import MlflowClient

from parameters import MODEL_NAME, IMAGE_SIZE, LATEST_VERSION_MODEL_ALIAS


def log_model_as_onnx(model, make_current=True):
    device = next(model.parameters()).device

    run_id = mlflow.active_run().info.run_id
    onnx_dir = "/tmp/medical-image-processing"
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = f"{onnx_dir}/hybrid_model_{run_id}.onnx"

    dummy_input = torch.randn(1, 3, *IMAGE_SIZE).to(device)

    print(f'Сохраняем модель в формате ONNX: {onnx_path}')
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
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

    print(f'Загружаем модель в MLflow: {onnx_path}')
    log_result = mlflow.onnx.log_model(
        onnx_model=onnx.load(onnx_path),
        artifact_path="onnx_model",
        registered_model_name=MODEL_NAME,
        input_example={"input": dummy_input.cpu().numpy().astype(np.float32)},
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

