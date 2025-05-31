import io
import os
from typing import Dict

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import HTTPException
from scipy.special import softmax

from logger import logger
from worker.parameters import MODEL_PATH


class Model:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Файл модели {MODEL_PATH} не найден!")
            raise FileNotFoundError(f"Файл модели {MODEL_PATH} не найден")
        self.load_model()

    def load_model(self):
        logger.info(f"Загружаем модель из файла {MODEL_PATH}...")
        self.ort_session = ort.InferenceSession(MODEL_PATH)
        model_input = self.ort_session.get_inputs()[0]
        self.input_name = model_input.name
        self.model_input_size = (model_input.shape[2], model_input.shape[3])  # (height, width)
        logger.info(f"Модель успешно загружена. Ожидаемый размер входа: {self.model_input_size}")

    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        self._check_content_length(image_bytes)
        input_data = self._preprocess_image(image_bytes)
        pred = self._predict(input_data)
        return pred

    def _check_content_length(self, image_bytes: bytes) -> None:
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=413, detail="Файл слишком большой (макс. 10МБ)")

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = image.resize(self.model_input_size)

            # Преобразование для ONNX модели
            image_array = np.array(image, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image_array = (image_array - mean) / std
            return np.expand_dims(image_array.transpose(2, 0, 1), axis=0)
        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}")
            raise HTTPException(status_code=400, detail=f"Ошибка обработки изображения: {e}")

    def _predict(self, image_np: np.ndarray) -> Dict[str, any]:
        try:
            if image_np.dtype != np.float32:
                image_np = image_np.astype(np.float32)

            outputs = self.ort_session.run(None, {self.input_name: image_np})
            logits = outputs[0]
            probabilities = softmax(logits, axis=1)[0]

            return {
                "prediction": int(np.argmax(probabilities)),
                "probability": float(probabilities[np.argmax(probabilities)]),
                "probabilities": probabilities.tolist(),
                "logits": logits.tolist()
            }
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при выполнении предсказания: {e}")
