import os

MODEL_PATH = os.path.join("models", "model.onnx")
CLASS_NAMES = ['Норма', 'Пневмония']
S3_STORAGE_BUCKET = os.environ.get('MLFLOW_S3_BUCKET')
PROD_IMAGES_FOLDER = "prod_images"
