import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from logger import logger
from model import Model
from model_registry_tools import ensure_model_file_exists, init_mlflow, download_new_model_version
from parameters import CLASS_NAMES
from plot_tools import create_probability_plot

app = FastAPI(title="X-Ray Classification API")
init_mlflow()
ensure_model_file_exists()
model = Model()


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Конечная точка для классификации рентгеновских снимков"""
    try:
        logger.info(f"Получен запрос на классификацию от {file.filename}")

        file_bytes = await file.read()
        pred = model.predict(file_bytes)
        plot_base64 = create_probability_plot(pred['probabilities'])

        # Формируем детализированный ответ
        pneumonia_prob = pred['probabilities'][1]  # Вероятность пневмонии
        is_pneumonia = pred['prediction'] == 1
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


@app.get("/update_model_version")
async def update_model_version_endpoint() -> JSONResponse:
    try:
        logger.info("Запрос на обновление версии модели")
        download_new_model_version()
        model.load_model()
        return JSONResponse(content={
            "status": "success",
            "message": "Модель успешно обновлена"
        })
    except Exception as e:
        logger.error(f"Ошибка при обновлении модели: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка при обновлении модели")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Запуск сервера на http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
