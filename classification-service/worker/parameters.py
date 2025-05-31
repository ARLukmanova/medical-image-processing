import os

MODEL_DIR = "models"
MODEL_NAME = "xray-hybrid-classifier"
LATEST_MODEL_VERSION_ALIAS = "current"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
CLASS_NAMES = ['Норма', 'Пневмония']
MLFLOW_PORT = os.environ.get('MLFLOW_PUBLIC_PORT')
MLFLOW_IP = os.environ.get('PUBLIC_SERVER_IP')
MODEL_VERSION_FILENAME = "model_version.txt"
