import os

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_NAME = 'xray-hybrid-classifier'
USE_GPU = os.environ.get("USE_GPU", "False").lower() == "true"
EXPERIMENT_NAME = "Hybrid-Training-GPU" if USE_GPU else "Hybrid-Training-CPU"
LATEST_VERSION_MODEL_ALIAS = 'current'
