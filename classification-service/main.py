from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
from PIL import Image
import io
import onnxruntime as ort
from scipy.special import softmax
import logging
from typing import Dict
import uvicorn
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import mlflow

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="X-Ray Classification API")

# Конфигурация модели
MODEL_DIR = "models"
MODEL_NAME = "Hybrid"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
CLASS_NAMES = ['Норма', 'Пневмония']

# Создаем папку для моделей, если её нет
os.makedirs(MODEL_DIR, exist_ok=True)

# Загрузка модели
ort_session = None
model_input_size = None



def download_model_from_mlflow(model_name: str, model_alias: str, dst_path: str):
    """
    Скачивает модель из MLflow Model Registry по имени и алиасу.
    """
    public_server_ip = os.environ.get('PUBLIC_SERVER_IP')
    ml_flow_public_port = os.environ.get('MLFLOW_PUBLIC_PORT')
    ml_flow_uri = f"http://{public_server_ip}:{ml_flow_public_port}/"
    mlflow.set_tracking_uri(ml_flow_uri)
    logger.info(f"MLFlow URI: {ml_flow_uri}")
    try:
        model_uri = f"models:/{model_name}@{model_alias}"
        logger.info(f"Скачивание модели из MLflow: {model_uri}")
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path)
        logger.info(f"Модель успешно скачана в {dst_path}")
    except Exception as e:
        logger.error(f"Ошибка при скачивании модели из MLflow: {e}")
        raise FileNotFoundError(f"Не удалось скачать модель из MLflow: {e}")


try:
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Файл модели {MODEL_PATH} не найден. Загружаем из MLflow...")
        download_model_from_mlflow(
            model_name="xray-hybrid-classifier",
            model_alias="current",
            dst_path=MODEL_DIR
        )
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Файл модели {MODEL_PATH} не найден!")
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

    # Загружаем ONNX модель
    ort_session = ort.InferenceSession(MODEL_PATH)
    input_shape = ort_session.get_inputs()[0].shape
    model_input_size = (input_shape[2], input_shape[3])  # (height, width)
    logger.info(f"Гибридная модель успешно загружена. Ожидаемый размер входа: {model_input_size}")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Предобработка изображения для гибридной модели"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize(model_input_size)

        # Преобразование для ONNX модели
        image_array = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        return np.expand_dims(image_array.transpose(2, 0, 1), axis=0)
    except Exception as e:
        logger.error(f"Ошибка предобработки: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка обработки изображения: {e}")


def predict(image_np: np.ndarray) -> Dict[str, float]:
    """Выполнение предсказания гибридной моделью"""
    try:
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)

        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: image_np})
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


def create_probability_plot(probabilities: list) -> str:
    """Создает график вероятностей и возвращает его как base64"""
    plt.figure(figsize=(8, 4))

    labels = ['Норма', 'Пневмония']
    colors = ['green', 'red']

    plt.bar(labels, probabilities, color=colors)
    plt.ylabel('Вероятность')
    plt.title('Результат классификации гибридной моделью')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)

    # Сохраняем график в base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Конечная точка для классификации рентгеновских снимков"""
    try:
        logger.info(f"Получен запрос на классификацию от {file.filename}")

        # Чтение и проверка файла
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=413, detail="Файл слишком большой (макс. 10МБ)")

        # Предобработка и предсказание
        input_data = preprocess_image(contents)
        pred = predict(input_data)

        pneumonia_prob = pred['probabilities'][1]  # Вероятность пневмонии
        is_pneumonia = pred['prediction'] == 1

        # Создаем график
        plot_base64 = create_probability_plot(pred['probabilities'])

        # Формируем детализированный ответ
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "prediction": {
                "class": CLASS_NAMES[pred['prediction']],
                "probability": float(pred['probability']),
                "is_pneumonia": bool(is_pneumonia),
                "pneumonia_probability": float(pneumonia_prob),
                "normal_probability": float(pred['probabilities'][0]),
                "message": "Анализ выполнен гибридной моделью (CNN + ViT)"
            },
            "plot_image": plot_base64,
            "recommendation": {
                "action": "Рекомендуется консультация врача" if is_pneumonia else "Патологий не обнаружено",
                "urgency": "Срочно" if pneumonia_prob > 0.7 else "Планово" if pneumonia_prob > 0.3 else "Не требуется"
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Необработанная ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>X-Ray Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { padding: 10px; border-radius: 5px; }
                .success { background-color: #d4edda; color: #155724; }
                .error { background-color: #f8d7da; color: #721c24; }
                form { margin-top: 20px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                input[type="file"] { margin-bottom: 10px; }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                input[type="submit"]:hover { background-color: #45a049; }
                .preview { max-width: 300px; margin-top: 20px; }
                .plot-image { max-width: 100%; margin-top: 20px; }
                .recommendation { margin-top: 15px; padding: 10px; border-left: 4px solid #007bff; }
                .urgency-high { color: #dc3545; font-weight: bold; }
                .urgency-medium { color: #ffc107; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Анализ рентгеновских снимков</h1>
                <h2>Гибридная модель (ResNet + EfficientNet)</h2>

                <form action="/predict" method="post" enctype="multipart/form-data" id="uploadForm">
                    <input type="file" name="file" accept="image/*" required>
                    <br>
                    <input type="submit" value="Анализировать снимок">
                </form>

                <div id="result" style="margin-top: 20px;"></div>

                <script>
                    document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                        e.preventDefault();
                        const formData = new FormData(this);
                        const resultDiv = document.getElementById('result');

                        resultDiv.innerHTML = '<p>Анализ снимка...</p>';

                        try {
                            const response = await fetch('/predict', {
                                method: 'POST',
                                body: formData
                            });

                            const data = await response.json();

                            if (data.status === 'success') {
                                const pred = data.prediction;
                                const rec = data.recommendation;

                                const urgencyClass = rec.urgency === 'Срочно' ? 'urgency-high' : 
                                                  rec.urgency === 'Планово' ? 'urgency-medium' : '';

                                resultDiv.innerHTML = `
                                    <div class="status success">
                                        <h3>Результат анализа:</h3>
                                        <p>Заключение: <strong>${pred.class}</strong></p>
                                        <p>Вероятность пневмонии: <strong>${(pred.pneumonia_probability * 100).toFixed(1)}%</strong></p>
                                        <p>Вероятность нормы: <strong>${(pred.normal_probability * 100).toFixed(1)}%</strong></p>
                                        <p><em>${pred.message}</em></p>

                                        <div class="recommendation">
                                            <h4>Рекомендации:</h4>
                                            <p>${rec.action}</p>
                                            <p class="${urgencyClass}">Срочность: ${rec.urgency}</p>
                                        </div>
                                    </div>
                                    ${data.plot_image ? `<img class="plot-image" src="data:image/png;base64,${data.plot_image}" alt="График вероятностей">` : ''}
                                `;
                            } else {
                                resultDiv.innerHTML = `
                                    <div class="status error">
                                        <p>Ошибка: ${data.detail || 'Неизвестная ошибка'}</p>
                                    </div>
                                `;
                            }
                        } catch (error) {
                            resultDiv.innerHTML = `
                                <div class="status error">
                                    <p>Ошибка при отправке запроса: ${error.message}</p>
                                </div>
                            `;
                        }
                    });
                </script>

                <h3>Инструкция по использованию:</h3>
                <ol>
                    <li>Загрузите рентгеновский снимок грудной клетки в формате JPG или PNG</li>
                    <li>Дождитесь обработки (обычно 10-20 секунд)</li>
                    <li>Получите заключение и рекомендации</li>
                </ol>

                <p><strong>Примечание:</strong> Данный анализ не заменяет консультацию врача.</p>
            </div>
        </body>
    </html>
    """


if __name__ == "__main__":
    # Конфигурация хоста и порта
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Запуск сервера на http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)